#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
    ProcessFailure,
    SignalException,
    Std,
)
from torch.distributed.elastic.utils.logging import get_logger

__all__ = ['WorkerSpec', 'Worker', 'WorkerState', 'WorkerGroup', 'RunResult', 'ElasticAgent', 'SimpleElasticAgent']
_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"

DEFAULT_ROLE = "default"
log = get_logger()


@dataclass
class WorkerSpec:
    """
    Contains blueprint information about a particular type of worker.
    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogenous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.

    Args:
        role: user-defined role for the workers with this spec
        local_world_size: number local workers to run
        fn: (deprecated use entrypoint instead)
        entrypoint: worker function or command
        args: arguments to pass to ``entrypoint``
        rdzv_handler: handles rdzv for this set of workers
        max_restarts: number of max retries for the workers
        monitor_interval: monitor status of workers every ``n`` seconds
        master_port: fixed port to run the c10d store on rank 0
                     if not specified then will chose a random free port
        master_addr: fixed master_addr to run the c10d store on rank 0
                     if not specified then will chose hostname on agent rank 0
        redirects: redirect std streams to a file,
                   selectively redirect for a particular
                   local rank by passing a map
        tee: tees the specified std stream(s) to console + file,
             selectively tee for a particular local rank by passing a map,
             takes precedence over ``redirects`` settings.

    """

    role: str
    local_world_size: int
    rdzv_handler: rdzv.RendezvousHandler
    fn: Optional[Callable] = None
    # TODO @kiuk - make entrypoint a required field
    entrypoint: Union[Callable, str, None] = None
    args: Tuple = ()
    max_restarts: int = 3
    monitor_interval: float = 30.0
    master_port: Optional[int] = None
    master_addr: Optional[str] = None
    local_addr: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE

    def __post_init__(self):
        assert self.local_world_size > 0
        assert self.monitor_interval > 0

        if self.fn:
            warnings.warn(
                "WorkerSpec.fn will be deprecated,"
                " please use WorkerSpec.entrypoint instead",
                category=DeprecationWarning,
            )
            self.entrypoint = self.fn
        assert self.entrypoint

    def get_entrypoint_name(self):
        """
        If the entrypoint is a function (e.g. ``Callable``) returns its ``__qualname__``,
        else if the entrypoint is a binary (e.g. ``str``), returns the binary name.
        """
        if isinstance(self.entrypoint, str):
            return os.path.basename(self.entrypoint)
        else:
            assert self.entrypoint is not None
            return self.entrypoint.__qualname__

'''
0x03 Worker
我们首先要看看 worker，这是 Agent 所管理的主体。

3.1 Worker 定义
Worker 类代表了一个worker实例，我们上文介绍了WorkerSpec，Worker 就是依据 WorkerSpec 构建出来的，其重点成员变量如下：

    id（任意）：唯一标识一个worker，具体是由ElasticAgent的特定实现来解释，对于本地代理，它可以是worker的pid（int），对于远程代理，它可以被编码为``host:port（string）`。
    
    local_rank ：worker的local rank。
    
    global_rank：worker的global rank。
    
    role_rank：具有相同角色的所有worker的rank。
    
    world_size：全局worker数量。
    
    role_world_size：具有相同角色的worker数量。
'''
class Worker:
    """
    Represents a worker instance. Contrast this with ``WorkerSpec`` that
    represents the specifications of a worker. A ``Worker`` is created from
    a ``WorkerSpec``. A ``Worker`` is to a ``WorkerSpec`` as an object is to
    a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Args:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """

    __slots__ = [
        "id",
        "local_rank",
        "global_rank",
        "role_rank",
        "world_size",
        "role_world_size",
    ]

    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        # unique identifier for this worker
        self.id: Any = None

        # rank of the worker among workers with the same role being monitored
        # by the same ``agent`` instance.
        self.local_rank: int = local_rank

        #  rank of the worker among all the workers across all roles
        #  across all ``agent`` instances.
        #  Global rank is not stable between re-rendezvous.
        self.global_rank: int = global_rank

        #  rank of the worker among all the workers with the same role
        #  across all ``agent`` instances.
        #  Role rank is not stable between re-rendezvous.
        self.role_rank: int = role_rank

        # total number of workers (globally). Due to elasticity
        # the world size may change between re-rendezvous.
        self.world_size: int = world_size

        # total number of workers that share the same role. Due to elasticity
        # the role world size may change between re-rendezvous.
        self.role_world_size: int = role_world_size

    def __str__(self):
        return (
            f"local_rank={self.local_rank},global_rank={self.global_rank}"
            f",role_rank={self.role_rank},world_size={self.world_size}"
            f",role_world_size={self.role_world_size}"
        )

    def __repr__(self):
        return str(self)

'''
3.3 WorkerState
WorkerState 表示 WorkerGroup的状态。工作组中的所有工作人员作为一个整体来维护/更改状态。如果工作组中的一个worker失败，则整个工作组被认为是失败：

  UNKNOWN - agent lost track of worker group state, unrecoverable
  INIT - worker group object created not yet started
  HEALTHY - workers running and healthy
  UNHEALTHY - workers running and unhealthy
  STOPPED - workers stopped (interruped) by the agent
  SUCCEEDED - workers finished running (exit 0)
  FAILED - workers failed to successfully finish (exit !0)
具体这些状态意义如下：

    UNKNOWN-代理丢失了对工作组状态的跟踪，无法恢复
    
    INIT-创建的工作组对象尚未启动
    
    HEALTHY-worker健康运行
    
    UNHEALTHY-worker在运行但是不健康
    
    STOPPED-代理停止（中断）worker
    
    SUCCEEDED-worker已完成运行（exit数值为0）
    
    FAILED-worker未能成功完成（exit数值不等于0)

工作组从初始的INIT状态开始，然后进入"健康"或"不健康"状态，最后到达终端"成功"或"失败"状态。工作组可以被代理打断并且临时置于"停止"状态。处于"已停止"状态的工作进程可以在不久的将来被调度重启，被设置为已停止的状态的例子为：

    观察到工作组故障|不健康
    检测到成员更改
当工作组上的操作（启动、停止、rdzv、重试等）失败，并导致操作部分应用于工作组时，状态将为"未知"。这通常发生在状态改变期间发生异常，而且异常未捕获/未处理的情况下。当工作组处于"未知"状态，代理不会恢复工作组，因此最好终止作业，并且由job manager重试节点。

WorkerState 具体定义如下：
'''
class WorkerState(str, Enum):
    """
    State of the ``WorkerGroup``. Workers in a worker group change state as a unit.
    If a single worker in a worker group fails the entire set is considered
    failed::

      UNKNOWN - agent lost track of worker group state, unrecoverable
      INIT - worker group object created not yet started
      HEALTHY - workers running and healthy
      UNHEALTHY - workers running and unhealthy
      STOPPED - workers stopped (interrupted) by the agent
      SUCCEEDED - workers finished running (exit 0)
      FAILED - workers failed to successfully finish (exit !0)


    A worker group starts from an initial ``INIT`` state,
    then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
    and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.

    Worker groups can be interrupted and temporarily put into ``STOPPED`` state
    by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
    in the near future by the agent. Some examples of workers being put into
    ``STOPPED`` state are:

    1. Worker group failure|unhealthy observed
    2. Membership change detected

    When actions (start, stop, rdzv, retry, etc) on worker group fails
    and results in the action being partially applied to the worker group
    the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
    exceptions during state change events on the agent. The agent is not
    expected to recover worker groups in ``UNKNOWN`` state and is better off
    self terminating and allowing the job manager to retry the node.
    """

    UNKNOWN = "UNKNOWN"
    INIT = "INIT"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        """
        Returns:
             True if the worker state represents workers still running
             (e.g. that the process exists but not necessarily healthy).
        """
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}

#WorkerGroup 代表了一个工作组。WorkerGroup 作为一个整体来管理多个 workers，进行批量处理。
class WorkerGroup:
    """
    Represents the set of ``Worker`` instances for the given ``WorkerSpec``
    managed by ``ElasticAgent``. Whether the worker group contains cross
    instance workers or not depends on the implementation of the agent.
    """

    __slots__ = ["spec", "workers", "store", "group_rank", "group_world_size", "state"]

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None
        self.group_rank = None
        self.group_world_size = None

        self.state = WorkerState.INIT

'''
4.3 ranks相关
前面的 _assign_worker_ranks 为 worker 建立 ranks，但是其内部有些细节我们还需要梳理一下。

4.3.1 _RoleInstanceInfo
这里要介绍一下 _RoleInstanceInfo 这个数据结构。代理使用该类与其他代理交换信息。该信息用于确定本代理workers的rank。这些代理工作在异构环境下，不同代理也许有不同数量的workers。其构建参数是：

    role (str) : 用户定义的role。
    rank (int) : 代理的rank。
    local_world_size (int) : 本地 workers 的数目。
'''
class _RoleInstanceInfo:
    """
    The class is used by the agent to exchange the information with other agents.
    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """

        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)


@dataclass
class RunResult:
    """
    Results returned by the worker executions. Run results follow an "all-or-nothing" policy
    where the run is successful if and only if ALL local workers managed by this agent
    complete successfully.

    If the result is successful (e.g. ``is_failed() = False``) then the ``return_values``
    field contains the outputs (return values) of the workers managed by THIS agent mapped
    by their GLOBAL ranks. That is ``result.return_values[0]`` is the return value of
    global rank 0.

    .. note:: ``return_values`` are only meaningful for when the worker entrypoint
              is a function. Workers specified as a binary entrypoint do not canonically
              have a return value and the ``return_values`` field is meaningless and
              may be empty.

    If ``is_failed()`` returns ``True`` then the ``failures`` field contains the
    failure information, again, mapped by the GLOBAL rank of the worker that failed.

    The keys in ``return_values`` and ``failures`` are mutually exclusive, that is,
    a worker's final state can only be one of: succeeded, failed. Workers intentionally
    terminated by the agent according to the agent's restart policy, are not represented
    in either ``return_values`` nor ``failures``.
    """

    state: WorkerState
    return_values: Dict[int, Any] = field(default_factory=dict)
    failures: Dict[int, ProcessFailure] = field(default_factory=dict)

    def is_failed(self) -> bool:
        return self.state == WorkerState.FAILED


def _get_socket_with_port() -> socket.socket:
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            log.info("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())

'''
2.4 基类
基类ElasticAgent 是一个 Abstract Class，真正运行的代理都需要由此派生。
从 ElasticAgent 的注释可知，代理进程负责管理一个或多个worker 进程。
工作进程被假定为常规分布式PyTorch脚本。
当worker进程由代理创建时，代理将为worker进程提供必要的信息，以便正确初始化torch进程组。
部署时，精确的拓扑和 agent-to-worker 比率取决于代理的具体实现和用户作业放置偏好。

ElasticAgent 有两个派生类：

    SimpleElasticAgent 实现了基类的部分函数，其目的是为了方便扩展新代理的实现。
    LocalElasticAgent 派生了SimpleElasticAgent ，是目前弹性训练最终使用的代理，主要用于在本地进行操作，负责管理单机上所有的worker进程。
'''
class ElasticAgent(abc.ABC):
    """
    Agent process responsible for managing one or more worker processes.
    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        log.exception(f"worker 0 failed with exit code : {failure.exit_code}")
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """
        Runs the agent, retrying the worker group on failures up to
        ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """
        Returns:
            The ``WorkerGroup`` for the given ``role``.
            Note that the worker group is a mutable object and hence in a
            multi-threaded/process environment it may change state.
            Implementors are encouraged (but not required) to return
            a defensive read-only copy.
        """
        raise NotImplementedError()

#在SimpleElasticAgent 初始化之中，会建立一个 WorkerGroup。
'''
具体如下：

+-----------------------------+      +------------------------------------------------+
| LocalElasticAgent           |      | WorkerSpec                                     |
|                             |      |                                                |
| +------------------------+  |      |   rdzv_handler = {DynamicRendezvousHandler} -------+
| |WorkerGroup             |  |      |                                                |   |
| |            spec +--------------> |   entry = worker_fn                            |   |
| |            workers     |  |      |                                                |   |
| |            store       |  |      |   role = {str} 'trainer'                       |   |
| |            group_rank  |  |      |                                                |   |
| |       group_world_size |  |      +------------------------------------------------+   |
| |                        |  |                                                           |
| +------------------------+  |                                                           |
|                             |                                                           |
| rdzv_run_id                 |                                                           |
| store                       |            +-----------------------------------------+    |
|                             |            |DynamicRendezvousHandler                 |    |
+-----------------------------+            |                                         |    |
                                           |                                         |    |
                                           |   _settings: RendezvousSettings         | <--+
                                           |                                         |
                                           |   _store: Store                         |
                                           |                                         |
                                           |   _state_holder: _RendezvousStateHolder |
                                           |                                         |
                                           |   _op_executor: _RendezvousOpExecutor   |
                                           |                                         |
                                           +-----------------------------------------+
                                           
                                           
在SimpleElasticAgent 初始化之中，会建立一个 WorkerGroup。   


0x04 SimpleElasticAgent
SimpleElasticAgent 是 Agent 的实现类之一。此抽象是为了方便扩展新的 agent 实现。
从后面可知，目前内置的 LocalElasticAgent 负责管理单机上的所有 worker 进程，如果用户希望只用一个代理就管理多机上所有的 worker，而不仅仅是本机 worker，那么可以通过扩展 SimpleElasticAgent 来实现一个自定义 Agent。                                        
'''
class SimpleElasticAgent(ElasticAgent):
    """
    An ``ElasticAgent`` that manages workers (``WorkerGroup``)
    for a single ``WorkerSpec`` (e.g. one particular type of worker role).
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._total_execution_time = 0

    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        return self._worker_group

    #调用 派生类的 _start_workers 来启动 worker 进程，因此基类这里没有实现，我们后续会看到派生类如何实现。
    '''
    目前逻辑如下，具体是：

        调用 rdzv_handler.next_rendezvous 来与其他 Node 进行同步。
        rdzv_handler.next_rendezvous 返回 ranks 等信息给_assign_worker_ranks。
        _assign_worker_ranks会生成一些Workers，其中每个 Worker都被自动分配了 rank。这些 workers 被 Agent的worker_group.workers所指向。
        +--------------------------------------------------+
        | LocalElasticAgent                                |         _initialize_workers
        |                                                  |                 +
        |                                                  |                 |
        |                                                  |                 |
        |   +----------------------+                       |                 v
        |   |WorkerGroup           |                       |         _rendezvous(worker_group)
        |   |                      |                       |                 +
        |   |     spec             |                       |                 |
        |   |                      |                       |                 | 1
        |   |     group_world_size |                       |                 v
        |   |                      |                       |        rdzv_handler.next_rendezvous()
        |   |     store            |                       |                 +
        |   |                      |    +----------------+ |                 |
        |   |     group_rank       |    | Worker0(rank 0)| |               2 | ranks
        |   |                      |    | Worker1(rank 1)| |  Workers        v
        |   |     workers  +----------> | ...            | | <----+ _assign_worker_ranks
        |   |                      |    | Workern(rank n)| |    3
        |   +----------------------+    +----------------+ |
        |                                                  |
        +--------------------------------------------------+
    '''
    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        r"""
        Starts ``worker_group.spec.local_world_size`` number of workers
        according to worker spec for the worker group .

        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        r"""
        Stops all workers in the given worker group. Implementors
        must deal with workers in all states defined by ``WorkerState``.
        That is, it must gracefully handle stopping non-existent workers,
        unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        r"""
        Checks on the workers for the ``worker_group`` and returns
        the new state of the worker group.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None:
        """
        Cleans up any resources that were allocated during the agent's work.

        Args:
            death_sig: Signal to send to the child process, SIGTERM is default
        """
        raise NotImplementedError()

    @staticmethod
    def _set_master_addr_port(
        store: Store,
        master_addr: Optional[str],
        master_port: Optional[int],
        local_addr: Optional[str],
    ):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        if master_addr is None:
            # If user specified the address for the local node, use it as the master addr if not exist
            if local_addr:
                master_addr = local_addr
            else:
                master_addr = _get_fq_hostname()

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    @staticmethod
    def _get_master_addr_port(store: Store) -> Tuple[str, int]:
        master_addr = store.get("MASTER_ADDR").decode(encoding="UTF-8")
        master_port = int(store.get("MASTER_PORT").decode(encoding="UTF-8"))
        return (master_addr, master_port)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    '''
    4.2.1 _rendezvous
我们首先看看_rendezvous，其做如下操作：

调用 next_rendezvous() 来处理成员关系变化，其会返回 world size，store等。
会把 store 配置到 workgroup 之中，后续worker 之间就可以通过这个kvstore进行沟通。
调用 _assign_worker_ranks 会生成 worker，并且为 worker 建立 ranks，返回的 workers 都赋值在代理的 worker_group.workers 之中。
以上两点都是利用 rendezvous 的信息来进行处理，比如从 rendezvous 之中提取 ranks。
    '''
    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""
        Runs rendezvous for the workers specified by worker spec.
        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """

        spec = worker_group.spec

        # 处理成员关系变化，注意，这里得到的是 group rank!
        store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
        self._store = store # store被设置到 Agent之中，store可以被认为是远端KV存储

        # 依据 group rank 为 worker 建立 ranks
        workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size

        if group_rank == 0:
            self._set_master_addr_port(
                store,
                spec.master_addr,
                spec.master_port,
                spec.local_addr,
            )

        master_addr, master_port = self._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        log.info(
            f"[{spec.role}] Rendezvous complete for workers. Result:\n"
            f"  restart_count={restart_count}\n"
            f"  master_addr={master_addr}\n"
            f"  master_port={master_port}\n"
            f"  group_rank={group_rank}\n"
            f"  group_world_size={group_world_size}\n"
            f"  local_ranks={[worker.local_rank for worker in workers]}\n"
            f"  role_ranks={[worker.role_rank for worker in workers]}\n"
            f"  global_ranks={[worker.global_rank for worker in workers]}\n"
            f"  role_world_sizes={[worker.role_world_size for worker in workers]}\n"
            f"  global_world_sizes={[worker.world_size for worker in workers]}\n"
        )

    '''
    依据 role infos 来确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。
    偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。workers 的等级为：[offset, offset+local_world_size]。
    
    目前逻辑拓展如下：

调用 rdzv_handler.next_rendezvous() 来和其他节点进行同步，获得信息。
获得信息中的store（可以认为就是远端的KV存储），group_world_size，group_rank 传给 Agent。
ranks 等信息传给 _assign_worker_ranks方法。
_assign_worker_ranks 之中，调用 _share_and_gather 在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共KV存储。
依据 role infos 来确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。
使用各种信息建立一系列的 Workers。
Workers 被复制给 Agent 的 WorkerGroup 之中。
                                                              _initialize_workers
                                                                      +
                                                                      |
                                                                      |
                                                                      v
                                                              _rendezvous(worker_group)
                                                                      +
+----------------------------------------------+                      |
| LocalElasticAgent                            |                      | 1
|                                              |   2                  v
|                                         +--------------+  rdzv_handler.next_rendezvous()
| +--------------------+                  |    |                      +
| | WorkerGroup        |                  |    |                      |
| |                    |                  |    |                    3 | ranks
| |                    |                  |    |                      v
| |  spec              |                  |    |       +--------------+------------------+
| |                    |                  |    |       | _assign_worker_ranks            |
| |                    |                  |    |       |                                 |
| |  store   <----------------------------+    |       |                        4        |
| |                    |                  |    |       | role_infos = _share_and_gather( |
| |                    |                  |    |       |               +          store) |
| |  group_world_size<--------------------+    |       |               | 5               |
| |                    |                  |    |       |               |                 |
| |                    |                  |    |       |               v                 |
| |  group_rank <-------------------------+    |       |          _get_ranks(world...)   |
| |                    |                       |       |          _get_ranks(role...)    |
| |                    |   +----------------+  |       |               +                 |
| |  workers  +----------->+ Worker0(rank 0)|  |       |               |                 |
| |                    |   | Worker1(rank 1)|  |       |               | 6               |
| |                    |   | ...            |  |Workers|               v                 |
| |                    |   | Workern(rank n)+<------------+ new Worker(local_rank,       |
| +--------------------+   +----------------+  |    7  |               global_rank,      |
|                                              |       |               role_rank,        |
+----------------------------------------------+       |               world_size,       |
                                                       |               role_world_size)  |
                                                       |                                 |
                                                       +---------------------------------+
_rendezvous 操作之后，Worker 实例已经生成了，接下来就看看如何生成 Worker 进程。但是因为这些方法在 SimpleElasticAgent 之中并没有实现，所以我们需要在其派生类 LocalElasticAgent 分析小节才能继续拓展我们的逻辑图。
    '''
    def _get_ranks(
        self,
        role_infos: List[_RoleInstanceInfo],
        role_idx: int,
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> Tuple[int, List[int]]:
        if end_idx == -1:
            end_idx = len(role_infos)
        prefix_sum = 0
        total_sum = 0
        for idx in range(start_idx, end_idx):
            if role_idx > idx:
                prefix_sum += role_infos[idx].local_world_size
            total_sum += role_infos[idx].local_world_size
        return (
            total_sum,
            list(range(prefix_sum, prefix_sum + role_infos[role_idx].local_world_size)),
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    '''
    4.2.3.2 为 worker 分配 ranks
    接着是调用 _assign_worker_ranks 为 worker 建立 ranks。分配 rank 算法如下：
    
        每个代理将其配置（group_rank, group_world_size , num_workers）写入公共存储。
        每个代理检索所有代理的配置，并使用角色和rank执行两级排序。
        确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。workers 的等级为：[offset, offset+local_world_size]。
        确定role rank：使用第3点中的算法确定role rank，不同之处是：偏移量计算是从与当前角色相同且具有最小 group rank 的第一个代理开始。
        因为所有代理都使用同样算法，所以其计算出的 ranks 数组都是相同的。
    然后生成 workers，把 worker 都赋值在 worker_group.workers 之中。
    '''
    @prof
    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> List[Worker]:
        """
        Determines proper ranks for worker processes. The rank assignment
        is done according to the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. Each agent retrieves configuration for all agents
           and performs two level sort using role and rank.
        3. Determine the global rank: the global rank of the workers for the current
           agent is the offset of the infos array up to group_rank of the agent.
           The offset is computed as a sum of local_world_size of all agents that
           have rank less than the group_rank. The workers would have the ranks:
           [offset, offset+local_world_size)
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the offset is done from the first
           agent that has the same role as current one and has the minimum group rank.
        """
        # 每个代理将其配置（group_rank, group_world_size, num_workers）写入公共存储。
        role_infos = self._share_and_gather(store, group_rank, group_world_size, spec)

        # 每个代理检索所有代理的配置，并使用角色和rank执行两级排序。
        my_role_info = role_infos[group_rank]

        # 确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。
        # 偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。workers 的等级为：[offset, offset+local_world_size]。
        worker_world_size, worker_global_ranks = self._get_ranks(role_infos, group_rank)
        role_infos = sorted(
            role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare)
        )
        role_start_idx, role_end_idx = _RoleInstanceInfo.find_role_boundaries(
            role_infos, my_role_info.role
        )
        role_pos = next(
            idx
            for idx, role_info in enumerate(role_infos)
            if _RoleInstanceInfo.compare(role_info, my_role_info) == 0
        )

        # 确定role rank：使用第3点中的算法确定role rank，不同之处是：偏移量计算是从与当前角色相同且具有最小 group rank 的第一个代理开始。
        role_world_size, role_ranks = self._get_ranks(
            role_infos, role_pos, role_start_idx, role_end_idx + 1
        )

        # 生成 workers，把 worker 都赋值在 worker_group.workers 之中。
        workers = []
        for ind in range(spec.local_world_size):
            worker = Worker(
                local_rank=ind,
                global_rank=worker_global_ranks[ind],
                role_rank=role_ranks[ind],
                world_size=worker_world_size,
                role_world_size=role_world_size,
            )
            workers.append(worker)
        return workers

    '''
    _share_and_gather 的作用是在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共存储。
    这里就是使用之前 Rendezvous 返回的 store 来进行信息共享。
    '''
    def _share_and_gather(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> List:
        agent_role_info = _RoleInstanceInfo(
            spec.role, group_rank, spec.local_world_size
        )
        key_prefix = "torchelastic/role_info"
        agent_config_enc = agent_role_info.serialize()
        role_infos_bytes = store_util.synchronize(
            store, agent_config_enc, group_rank, group_world_size, key_prefix
        )
        role_infos = [
            _RoleInstanceInfo.deserialize(role_info_bytes)
            for role_info_bytes in role_infos_bytes
        ]
        return role_infos

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    '''
    4.2 初始化workers
        代理主循环之中，首先使用 self._initialize_workers(self._worker_group) 来启动 worker。在 _initialize_workers之中：
        
        首先使用 self._rendezvous(worker_group) 进行节点之间的同步共识操作以及rank处理等等。
        其次调用 _start_workers 启动 workers。这里的 _start_workers 是虚函数，需要派生类实现。
    '''
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""
        Starts a fresh set of workers for the worker_group.
        Essentially a rendezvous followed by a start_workers.

        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        log.info(f"[{role}] Rendezvous'ing worker group")

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group) # 同步共识操作  # Worker实例已经生成了

        log.info(f"[{role}] Starting worker group")
        worker_ids = self._start_workers(worker_group) # 启动worker
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id  # 得到进程ID

        worker_group.state = WorkerState.HEALTHY

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    # _restart_workers 是重启 workers。
    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """
        Restarts (stops, rendezvous, starts) all local workers in the group.
        """

        role = worker_group.spec.role
        log.info(f"[{role}] Stopping worker group")
        self._stop_workers(worker_group)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    # SimpleElasticAgent 是 LocalElasticAgent 的基类，所以会先运行到WorkerSpec.run 方法这里，run方法则调用了 _invoke_run。
    @prof
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        start_time = time.monotonic()
        shutdown_called: bool = False
        try:
            result = self._invoke_run(role)   # 调用
            self._total_execution_time = int(time.monotonic() - start_time)
            self._record_metrics(result)
            self._record_worker_events(result)
            return result
        except SignalException as e:
            log.warning(f"Received {e.sigval} death signal, shutting down workers")
            self._shutdown(e.sigval)
            shutdown_called = True
            raise
        finally:
            if not shutdown_called:
                self._shutdown()
            # record the execution time in case there were any exceptions during run.
            self._total_execution_time = int(time.monotonic() - start_time)

    def get_event_failed(self) -> Event:
        return self._construct_event(
            state="FAILED",
            source=EventSource.AGENT,
            raw_error=traceback.format_exc(),
        )

    def get_event_succeeded(self) -> Event:
        return self._construct_event(
            state="SUCCEEDED",
            source=EventSource.AGENT,
        )

    def _record_worker_events(self, result: RunResult) -> None:
        for worker in self._worker_group.workers:
            failure = result.failures.get(worker.global_rank)
            state: str = self._get_worker_state(worker, result)
            raw_error = json.dumps(failure.error_file_data) if failure else None
            record(self._construct_event(state, EventSource.WORKER, worker, raw_error))

    def _get_worker_state(self, worker: Worker, result: RunResult) -> str:
        failure = result.failures.get(worker.global_rank)
        if result.state in {WorkerState.UNHEALTHY, WorkerState.FAILED} and not failure:
            # The worker got terminated by the torchelastic agent via SIGTERM signal
            return "TERMINATED"
        elif failure or worker.global_rank in result.return_values:
            return result.state.value
        else:
            raise ValueError(f"Unknow worker: {worker.global_rank}")

    def _construct_event(
        self,
        state: str,
        source: EventSource,
        worker: Optional[Worker] = None,
        raw_error: Optional[str] = None,
    ) -> Event:
        wg = self._worker_group
        spec = wg.spec
        md = {
            "group_world_size": wg.group_world_size,
            "entry_point": spec.get_entrypoint_name(),
        }
        if worker:
            md["local_rank"] = (worker.local_rank,)
            md["role_rank"] = (worker.role_rank,)
            md["role_world_size"] = (worker.role_world_size,)
            global_rank = worker.global_rank
            worker_id = str(worker.id)
        else:
            global_rank = None
            worker_id = None
        md_str = json.dumps(md)
        metadata = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": global_rank,
            "group_rank": wg.group_rank,
            "worker_id": worker_id,
            "role": spec.role,
            "hostname": _get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": raw_error,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
        }
        return Event(
            f"torchelastic.worker.status.{state}", source=source, metadata=metadata
        )

    def _record_metrics(self, group_results: RunResult):
        is_failed = group_results.is_failed()
        self._record_flakiness_metric(is_failed)
        spec = self._worker_group.spec
        restarts_happened = self._remaining_restarts != spec.max_restarts
        put_metric(f"workers.{spec.role}.run_total", 1)
        self._record_metric_with_condition(
            "run_success_with_retries", not is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not is_failed and not restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_with_retries", is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", is_failed and not restarts_happened
        )

    def _record_metric_with_condition(self, metric_name, condition):
        spec = self._worker_group.spec
        if condition:
            put_metric(f"workers.{spec.role}.{metric_name}", 1)
        else:
            put_metric(f"workers.{spec.role}.{metric_name}", 0)

    def _record_flakiness_metric(self, is_failed: bool = False):
        if is_failed:
            flakiness = 100.0
        else:
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        spec = self._worker_group.spec

        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))

    '''
    4.5 代理主循环
    代理在 invoke_run 之中做如下操作：
    
    启动 _initialize_workers，这里会使用 _rendezvous 构建一个 rendezvous，然后调用 _start_workers 启动 workers。
    进入 while True 循环，在循环之中：
        通过 _monitor_workers 定期轮训用户程序运行情况，得到客户进程运行结果，然后依据情况作出判断。
            如果程序正常结束，则返回。
            如果程序出错，则重试，即重启所有 workers，如果重试次数达到依然有问题，就结束所有workers。
            如果节点成员关系有变化，比如scale up就会有新的节点在waiting，这时候就重启所有workers。
            
            
于是最终逻辑如下：

+----------------------------------------------+
| LocalElasticAgent                            |
|                                              |    +---------------------------------------------------+
|  rdzv_run_id                                 |    | WorkerSpec                                        |
|                                              |    |                                                   |
|  store           +------------------------+  |    |      rdzv_handler = {DynamicRendezvousHandler} +-------+
|                  |WorkerGroup             |  |    |                                                   |    |
|  _pcontext       |            spec +------------> |      entry = worker_fn                            |    |
|                  |            workers     |  |    |                                                   |    |
|                  |            store       |  |    |      role = {str} 'trainer'                       |    |
|                  |            group_rank  |  |    |                                                   |    |
|                  |       group_world_size |  |    +---------------------------------------------------+    |
|                  |                        |  |                                                             |
|                  +------------------------+  |                                                             |
|  +----------------------------------------+  |                                                             |
|  | _invoke_run                            |  |                                                             |
|  |                                        |  |             +-----------------------------------------+     |
|  |   _initialize_workers +------------------------+        |DynamicRendezvousHandler                 |     |
|  |                                        |  |    |        |                                         |     |
|  |                                        |  |    |        |                                         |     |
|  |   while True:                          |  |    |        |   _settings: RendezvousSettings         | <---+
|  |       _monitor_workers(_worker_group)  |  |    |        |                                         |
|  |                +                       |  |    |        |   _store: Store                         |
|  |                | _pcontext.wait        |  |    |        |                                         |
|  |                |                       |  |    |        |   _state_holder: _RendezvousStateHolder |
|  +----------------------------------------+  |    |        |                                         |
|                   |                          |    |        |   _op_executor: _RendezvousOpExecutor   |
+----------------------------------------------+    |        |                                         |
                    |                               |        +-----------------------------------------+
                    |                               |
                    v                               v
         +-------------------------------------------------+
         |  +------------+  +------------+  +------------+ |
         |  |Process     |  |Process     |  |Process     | |
         |  |            |  |            |  |            | |
         |  |    work_fn |  |   work_fn  |  |    work_fn | |
         |  |            |  |            |  |            | |
         |  +------------+  +------------+  +------------+ |
         +-------------------------------------------------+

4.1 总体运行
SimpleElasticAgent 主循环 _invoke_run 是核心逻辑（这里默认代理和worker在同一个机器之上），其中做如下操作：

    使用 self._initialize_workers(self._worker_group) 完成初始化工作，比如来启动 worker，为每个worker 分配 rank 等等。
    然后进入 while True 循环，在循环之中通过 _monitor_workers 定期轮训用户程序运行情况，得到 worker 进程运行结果，然后依据情况进行不同处理。
        如果程序正常结束，则返回。
        如果程序出错，则重试，如果重试次数达到，结束workers。
        如果节点成员关系有变化，比如scale up就会有新的节点在waiting，这时候就重启所有workers。

    '''

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        log.info(
            f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()}"
        )

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            # 定期监控
            time.sleep(monitor_interval)
            # 监控客户程序运行情况
            run_result = self._monitor_workers(self._worker_group) # 得到进程运行结果
            state = run_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                # 程序正常结束
                log.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
                )
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                # 程序出错
                if self._remaining_restarts > 0: # 重试
                    log.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                        f" will restart worker group"
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)  # 重试次数达到，结束workers
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                # 节点成员关系有变化，比如scale up，就会有新节点waiting
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank

                # 如果有新的节点在waiting，就重启所有workers
                if num_nodes_waiting > 0:
                    log.info(
                        f"[{role}] Detected {num_nodes_waiting} "
                        f"new nodes from group_rank={group_rank}; "
                        f"will restart worker group"
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    #实际上，几乎不可能保证DDP的所有 worker 都能保证同时结束，所以因此TE提供了一个finalization barrier，这个barrier的作用是对worker finalization 实施等待超时（5分钟）。
    def _exit_barrier(self):
        """
        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times. This barrier keeps the agent process alive until all workers finish.
        """
        log.info(
            f"Local worker group finished ({self._worker_group.state}). "
            f"Waiting {self._exit_barrier_timeout} seconds for other agents to finish"
        )
        start = time.time()
        try:
            store_util.barrier(
                self._store,
                self._worker_group.group_rank,
                self._worker_group.group_world_size,
                key_prefix=_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            log.info(
                f"Done waiting for other agents. Elapsed: {time.time() - start} seconds"
            )
        except SignalException as e:
            log.warning(f"Got termination signal: {e.sigval}")
            raise
        except Exception:
            log.exception(
                f"Error waiting on exit barrier. Elapsed: {time.time() - start} seconds"
            )
