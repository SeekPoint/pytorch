# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
    NodeState,
    construct_and_record_rdzv_event,
)

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousTimeoutError,
)
from .utils import _delay, _PeriodicTimer

__all__ = ['RendezvousBackend', 'RendezvousTimeout', 'RendezvousSettings', 'DynamicRendezvousHandler', 'create_handler']

log = logging.getLogger(__name__)


def get_method_name(depth=2):
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return "no_method_name"


Token = Any
"""Represents an opaque fencing token used by the rendezvous backend."""
'''
3.5 后端
在 PyTorch 之中，backend 概念指的是当前进程要使用的通信后端，一般来说，支持的通信后端有 gloo，mpi，nccl 。建议用 nccl。

在弹性训练这里，DynamicRendezvousHandler 需要我们在构建时候指定后端（RendezvousBackend）。用户可以自己实现后端，或者使用如下PyTorch附带实现之一:

C10dRendezvousBackend，其使用 C10d 存储（默认是 TCPStore） 作为 rendezvous backend，其优势是不需要依赖第三方，比如etcd，来构建一个rendezvous 。
EtcdRendezvousBackend，其使用EtcdRendezvousHandler，EtcdRendezvousBackend 等类来基于 etcd 完成。
因为 EtcdRendezvousBackend 必须依赖 ETCD，需要安装一个 ETCD集群，所以推荐使用 c10d 后端，其易用性更好。我们接下来就主要介绍 c10d 后端。

C10d 后端主要基于一个 TCPStore，通过 TCP 进行同步。我们在之前文章中介绍过 TCPStore，TCPStore 是基于 TCP 的分布式键值存储实现（类似于 Redis）。是一个典型的 client-server 架构，服务器存储/保存数据，而存储客户端可以通过 TCP 连接到服务器存储并执行诸如set()插入键值对、get()检索键值对等操作。

所以，对于 c10d 后端来说，在其中一个代理之上会运行 TCPStore Master，其负责监听端口，提供API，Rendezvous 的各种同步操作，都是由各个代理连接到这个中心化的 TCPStore Master，在其上完成。

具体可以如下图所示，来源于知乎 BobLiu。

IMG！！！！有图！！！

3.5.1 使用
下图展示了如何配置后端

     store = TCPStore("localhost")
     backend = C10dRendezvousBackend(store, "my_run_id") # 配置了后端

     rdzv_handler = DynamicRendezvousHandler.from_backend(
         run_id="my_run_id",
         store=store,
         backend=backend,
         min_nodes=2,
         max_nodes=4
     )
3.5.2 基类
我们首先看看后端的基类 RendezvousBackend。这是一个虚类，主要功能就是设置和获取State。

'''
class RendezvousBackend(ABC):
    """Represents a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gets the name of the backend."""

    @abstractmethod
    def get_state(self) -> Optional[Tuple[bytes, Token]]:
        """Gets the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Optional[Token] = None
    ) -> Optional[Tuple[bytes, Token, bool]]:
        """Sets the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """


class RendezvousTimeout:
    """Holds the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        keep_alive:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """

    _ZERO = timedelta(0)

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
        "heartbeat": timedelta(seconds=5),
    }

    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(
        self,
        join: Optional[timedelta] = None,
        last_call: Optional[timedelta] = None,
        close: Optional[timedelta] = None,
        heartbeat: Optional[timedelta] = None,
    ) -> None:
        self._set_timeouts(join=join, last_call=last_call, close=close, heartbeat=heartbeat)

    @property
    def join(self) -> timedelta:
        """Gets the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        """Gets the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        """Gets the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        """Gets the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: Optional[timedelta]):
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)

'''
RendezvousSettings 类用来存储rendezvous的配置。可以理解为静态元信息。

    run_id : rendezvous 的 id。
    min_nodes ：rendezvous 的最小节点数目。
    max_nodes ：rendezvous 的最大节点数目。
    timeout ：超时时间。
    keep_alive_interval ：节点在发送心跳之间等待的时间量。
    keep_alive_max_attempt ： 心跳的最大重试次数。
'''
@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    """Holds the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int

#_NodeDesc 是rendezvous的一个节点。
@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describes a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    """Generates node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0

    def generate(self, local_addr: Optional[str] = None) -> _NodeDesc:
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)

'''
3.3 状态
_RendezvousState 是rendezvous的状态。是动态信息，每一个 node 都会维护一个本地 state。

    round：Rendezvous的当前轮次
    
    complete：一个布尔值，指示rendezvous当前一轮是否完成了。
    
    deadline：截止时间，如果如果当前轮次一直在等待节点加入，如果这个参数设置了，就是等待的截至时间。
    
    closed：一个布尔值，指示rendezvous是否结束了。
    
    participants：字典结构，存放参与者和它们对应ranks。
    
    wait_list：set结构，存放等待参与下一轮rendezvous操作的一组节点
    
    last_heartbeats：字典，包含每个节点上次心跳时间。
'''
class _RendezvousState:
    """Holds the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """

    round: int
    complete: bool
    deadline: Optional[datetime]
    closed: bool
    participants: Dict[_NodeDesc, int]
    wait_list: Set[_NodeDesc]
    last_heartbeats: Dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.last_heartbeats = {}


def _remove_participant_epilogue(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            state.complete = False

            state.round += 1
    else:
        if len(state.participants) < settings.min_nodes:
            state.deadline = None

# 这个类的作用是保存与其他节点同步的rendezvous状态，但是需要一个派生类来完成功能。
class _RendezvousStateHolder(ABC):
    """Holds the shared rendezvous state synced with other nodes."""

    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        """Gets the local state."""

    @abstractmethod
    def sync(self) -> Optional[bool]:
        """Reads or writes the latest state.

        Returns:
            A boolean value indicating whether the local state, in case marked
            as dirty, was successfully synced with other nodes.
        """

    @abstractmethod
    def mark_dirty(self) -> None:
        """Marks the local state as dirty."""

#_BackendRendezvousStateHolder 拓展了_RendezvousStateHolder。其 sync 就是调用内部的 后端，对 store 进行读写。
class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Holds the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """

    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: List[_NodeDesc]

    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        cache_duration: int = 1,
    ) -> None:
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING):
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
        )

    @property
    def state(self) -> _RendezvousState:
        """See base class."""
        return self._state

    #在 run 函数之中，需要注意的是：在执行各种算子操作之前，会调用 self._state_holder.sync() 在各个 worker 之间进行一个状态同步，达成共识 （consensus）。
    #torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py 之中是对应后端代码。
    def sync(self) -> Optional[bool]:
        """See base class."""
        state_bits: Optional[bytes] = None

        token = None

        has_set: Optional[bool]

        if self._dirty: # 如果本node状态变化了
            has_set = False

            state_bits = pickle.dumps(self._state)

            # 这里会对后端进行设置  # 把自己的状态设置到backend之中
            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:  # 自己没变化，只能从后端获取
            has_set = None

            if self._cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time >= max(time.monotonic() - self._cache_duration, 0):
                    return None

            get_response = self._backend.get_state()  # 从backend获取其他节点最新状态
            if get_response is not None:
                state_bits, token = get_response

        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits) # 用后端状态更新本身的状态
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            self._state = _RendezvousState()

        if has_set and self._dead_nodes and log.isEnabledFor(logging.DEBUG):
            node_list = ", ".join(f"'{dead_node}'" for dead_node in self._dead_nodes)

            msg = (
                f"As part of the sync operation the node(s) {node_list} have been removed from the "
                f"rendezvous '{self._settings.run_id}' since they had no heartbeat."
            )
            self._record(message=msg)
            log.debug(msg)

        self._token = token

        self._dirty = False

        self._last_sync_time = time.monotonic()

        self._sanitize()

        return has_set

    '''
    _sanitize 方法用来依据其他节点消息做处理，比如清理故障节点。即，如果上一次的心跳时间超过了一定阈值范围，则会把这些节点标记为dead_node，并且从 participant或者wait list中清除这些节点。
    '''
    def _sanitize(self) -> None:
        state = self._state

        expire_time = datetime.utcnow() - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]

        participant_removed = False

        for dead_node in self._dead_nodes:
            del state.last_heartbeats[dead_node]  # 移除故障节点

            try:
                del state.participants[dead_node]  # 移除故障节点

                participant_removed = True
            except KeyError:
                pass

            try:
                state.wait_list.remove(dead_node) # 移除故障节点
            except KeyError:
                pass

        if participant_removed:
            # Common epilogue shared with the _remove_from_participants()
            # function of _DistributedRendezvousOpExecutor.
            _remove_participant_epilogue(state, self._settings)

    def mark_dirty(self) -> None:
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """
        self._dirty = True

'''
0x03 算子
_RendezvousOpExecutor 引擎的业务逻辑被分成两层：用户操作 和 内部业务逻辑。用户操作和内部业务机制之间被解耦。

用户操作被分成各种算子，包括：心跳，Join，关闭，结束。比如Join 算子就是 _RendevzousJoinOp。

内部业务逻辑被分成各种业务函数，比如 _add_to_participants 方法从等待列表中移除节点，往 participants 加入这个节点。

算子和内部业务逻辑并不是一一对应，需要一个类似状态机的机制来控制。

比如，心跳操作算子的结果可能是：超时/keep alive/正常结束，所以应该根据这个结果调用不同的内部业务函数。这种对应关系逻辑就是通过 Action 来完成的。
各种算子联合起来，聚合成了一个状态机。
算子内部就是生成各种 Action，决定了状态机的下一步操作。
引擎内部就是根据 Action 来执行具体业务逻辑，或者可以说，是通过 Action 进行解耦。

具体如下，引擎从逻辑上可以分成三层：最上面是算子层，中间是 Action 层，下面是业务函数层。

+-----------------------------------------------------------------------------------------+
|                                                                                         |
| _RendezvousKeepAliveOp    _RendezvousCloseOp    _RendezvousExitOp    _RendezvousJoinOp  |
|                                                                                         |
+-------------+---------------------+--------------------+------------------+-------------+
              |                     |                    |                  |
              |                     |                    |                  |
              |                     |                    |                  |
              |                     |                    |                  |
              v                     v                    v                  v

+-----------------------------------------------------------------------------------------+
|                                                                                         |
| KEEP_ALIVE   ADD_TO_PARTICIPANTS   ADD_TO_WAIT_LIST   REMOVE_FROM_WAIT_LIST   ......    |
|                                                                                         |
+-------------+----------+----------+----------+---------+---------+---------+------------+
              |          |          |          |         |         |         |
              |          |          |          |         |         |         |
              |          |          |          |         |         |         |
              |          |          |          |         |         |         |
              v          v          v          v         v         v         v

+-----------------------------------------------------------------------------------------+
|                                                                                         |
| _add_to_participants    _remove_from_participants     _add_to_wait_list        ......   |
|                                                                                         |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
我们逐一解析。


'''

class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""

    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    REMOVE_FROM_PARTICIPANTS = 4
    REMOVE_FROM_WAIT_LIST = 5
    MARK_RENDEZVOUS_COMPLETE = 6
    MARK_RENDEZVOUS_CLOSED = 7
    SYNC = 8
    ERROR_CLOSED = 9
    ERROR_TIMEOUT = 10
    FINISH = 11

# 其作用是把 Rendezvous 的各种信息封装了起来，提供给操作引擎。
# 这里就有了 _RendezvousState 和 RendezvousSettings 的使用。
class _RendezvousContext:
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """

    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings

    def __init__(
        self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings
    ) -> None:
        self.node = node
        self.state = state
        self.settings = settings

'''
0x01 前言
1.1 总体系统
弹性训练可以理解为在 Rendezvous 基础之上的一个运行系统。



Agent 偏重具体节点上的逻辑

    Agent 负责具体业务逻辑相关操作，比如启动进程执行用户程序，监控用户程序运行情况，如果有异常就通知 Rendezvous。
    Agent 是一个 worker manager，负责启动/管理 workers 进程，组成一个 worker group，监控 workers 运行状态，捕获失效 workers，如果有故障/新加入worker，则重启 worker group。
    Agent负责维护 WORLD_SIZE 以及 RANK 信息。用户不需要再手动提供，Agent会自动处理这些。
    Agent 是具体节点上的后台进程，是独立个体。Agent自己无法实现整体上的弹性训练，所以需要一个机制来完成 worker 之间的相互发现，变更同步等等（WORLD_SIZE 和 RANK 这些信息其实也需要多个节点同步才能确定），这就是下面的 Rendezvous 概念。
Rendezvous 负责

集群逻辑

，保证节点之间对于""有哪些节点参与训练"达成强一致共识。

    每一个 Agent 内部包括一个 Rendezvous handler，这些 handler 总体上构成了一个 Rendezvous 集群，从而构成了一个 Agent 集群。
    Rendezvous 完成之后，会创建一个共享键值存储（shared key-value store），这个store实现了一个torch.distributed.Store API。此存储仅由已完成Rendezvous的成员共享，它旨在让Torch Distributed Elastic在初始化作业过程之中交换控制和数据信息。
    Rendezvous 负责在每个agent之上维护当前 group 所有相关信息。每个 agent 之上有一个 rendezvous，它们会互相通信，总体维护一套信息，这些信息存储在上面提到的Store 之中。
    Rendezvous 负责集群逻辑相关，比如新加入节点，移除节点，分配rank等等。
1.2 Rendezvous
目前为止，Rendezvous 信息如下，DynamicRendezvousHandler 属于动态逻辑，其中，_RendezvousStateHolder 是状态等元信息存储（静态结构），大家会发现图中还有一个 _RendezvousOpExecutor 没有介绍，这就是运行时引擎，所以我们本文看看 _RendezvousOpExecutor 如何处理。

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
1.3 解耦
_RendezvousOpExecutor 把功能分割解耦：

    业务逻辑被抽象成为一系列算子，比如 _RendevzousJoinOp。
    Rendezvous 内部维护了一套由业务函数组成的状态机，比如函数 _add_to_participants 用来添加参与者。
    _RendezvousOpExecutor 引擎来执行各种算子，依据算子结果，得到一个 Action，再利用 Action 调用业务函数进行操作。
本文主要介绍C10d 后端对应的 Rendezvous 引擎。


_RendezvousOpExecutor 是引擎的基类，只是定义了run这个虚函数。
'''
class _RendezvousOpExecutor(ABC):
    """Executes rendezvous operations."""

    @abstractmethod
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
    ) -> None:
        """Executes a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
        """

'''
2.2 分布式操作引擎
_DistributedRendezvousOpExecutor 拓展了 _RendezvousOpExecutor，是 ElasticTorch 的实际执行者。类似于 Looper，负责消息分发，调用业务，状态维护。

2.2.1 定义
与其基类相比，_DistributedRendezvousOpExecutor 加入了比如节点信息，状态，配置这样的成员变量。


逻辑如下：

+---------------------------------------------------------------+
| _DistributedRendezvousOpExecutor                              |
|                                                               |
|                     +------------------------+                |
|        _state +---> | _RendezvousState       |                |
|                     |                        |                |
|                     |       participants     |                |
|                     |       wait_list        |                |
|                     |       last_heartbeats  |                |
|                     |       deadline         |                |
|                     +------------------------+                |
|                                                               |
|                     +-------------------------+               |
|      _settings +--> | RendezvousSettings      |               |
|                     |                         |               |
|                     +-------------------------+               |
|                                                               |
|                     +--------------------------------------+  |
| _state_holder +---> | _BackendRendezvousStateHolder        |  |
|                     |                                      |  |
|                     |        _backend: RendezvousBackend   |  |
|                     |        _state: _RendezvousState      |  |
|                     |        _settings: RendezvousSettings |  |
|                     |                                      |  |
|                     +--------------------------------------+  |
|                     +--------------------------------------+  |
|                     | _NodeDesc                            |  |
|     _node +-------> |              fqdn: str               |  |
|                     |              pid: int                |  |
|                     |              local_id: int           |  |
|                     |                                      |  |
|                     +--------------------------------------+  |
+---------------------------------------------------------------+

'''
class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Executes rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """

    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings

    def __init__(
        self,
        node: _NodeDesc,
        state_holder: _RendezvousStateHolder,
        settings: RendezvousSettings,
    ) -> None:
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._node.addr,
            pid=self._node.pid,
            local_id=self._node.local_id,
        )
    '''
    StateHolder 具体如何使用在 _DistributedRendezvousOpExecutor 之中有（以下代码精简）：
        通过 _state_holder.sync() 同步各种状态，因为最新状态在 rendezvous。
        通过 self._state_holder.state 得到最新的状态。
        进行业务处理。
        通过 _state_holder.mark_dirty() 再次同步，把自己状态同步给其他节点
        
        
        
我们把目前逻辑总结如下，两个 _BackendRendezvousStateHolder 通过 TCPStore 进行信息交互。

                                                                       +
+-------------------------------+                                      |                                        +-------------------------------+
| _BackendRendezvousStateHolder |                                      |                                        | _BackendRendezvousStateHolder |
|                               |     +-------------------+            |           +--------------------+       |                               |
|             _settings +-----------> | RendezvousSettings|            |           | RendezvousSettings | <----------+ _settings                |
|                               |     +-------------------+            |           +--------------------+       |                               |
|                               |     +-------------------+            |           +--------------------+       |                               |
|             _state +--------------> | _RendezvousState  |            |           | _RendezvousState   | <----------+ _state                   |
|                               |     |                   |            |           |                    |       |                               |
|                               |     +-------------------+            |           +--------------------+       |                               |
|                               |                                      |                                        |                               |
|                               |     +-----------------------+        +           +----------------------+     |                               |
|             _backend +------------> | C10dRendezvousBackend |                    | C10dRendezvousBackend| <-------+  _backend                 |
|                               |     |                       |    +---------+     |                      |     |                               |
|                               |     |             _store +-----> |TCPStore | <---------+ _store         |     |                               |
|                               |     |                       |    |         |     |                      |     |                               |
|                               |     +-----------------------+    +---------+     +----------------------+     |                               |
|                               |                                                                               |                               |
|                               |         ^                            +                       ^                |                               |
|                               |         |                            |                       |                |                               |
|                               |         |                            |                       |                |                               |
|             sync +----------------------+                            |                       +---------------------+  sync                    |
|                               |   set_state                          |                         set_state      |                               |
+-------------------------------+                                      +                                        +-------------------------------+


run 函数实现了基础逻辑，就是依据 action 类型进行各种操作。

具体如下图。

+-----------------------------------------+                          +---------------------------------------------------------------+
|DynamicRendezvousHandler                 |                          | _DistributedRendezvousOpExecutor                              |
|                                         |                          |                                                               |
|                                         |                          |                     +------------------------+                |
|   _settings: RendezvousSettings         |                          |        _state +---> | _RendezvousState       |                |
|                                         |                          |                     |                        |                |
|                                         |                          |                     |       participants     |                |
|   _store: Store                         |                          |                     |       wait_list        |                |
|                                         |                          |                     |       last_heartbeats  |                |
|                                         |                          |                     |       deadline         |                |
|   _state_holder: _RendezvousStateHolder |                          |                     +------------------------+                |
|                                         | run(_RendezvousJoinOp()) |                     +-------------------------+               |
|                                         |                          |      _settings +--> | RendezvousSettings      |               |
|   _op_executor  +------------------------------------------------> |                     |                         |               |
|                                         |                          |                     +-------------------------+               |
|                                         |                          |                     +--------------------------------------+  |
+-----------------------------------------+                          | _state_holder +---> | _BackendRendezvousStateHolder        |  |
                                                                     |                     |                                      |  |
                                                                     |                     |        _backend: RendezvousBackend   |  |
                                                                     |                     |        _state: _RendezvousState      |  |
                                                                     |                     |        _settings: RendezvousSettings |  |
                                                                     |                     |                                      |  |
                                                                     |                     +--------------------------------------+  |
                                                                     |                     +--------------------------------------+  |
                                                                     |                     | _NodeDesc                            |  |
                                                                     |     _node +-------> |              fqdn: str               |  |
                                                                     |                     |              pid: int                |  |
                                                                     |                     |              local_id: int           |  |
                                                                     |                     |                                      |  |
                                                                     |                     +--------------------------------------+  |
                                                                     +---------------------------------------------------------------+
                                                                     
                                                                    
    '''

    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
    ) -> None:
        """See base class."""
        action = None

        while action != _Action.FINISH: # 循环，一直到获得一个FINISH action 为止
            # Reads or writes the latest rendezvous state shared by all nodes in
            # the rendezvous. Note that our local changes might get overridden
            # by another node if that node synced its changes before us.

            # 这里很重要，在所有node之间做信息同步
            has_set = self._state_holder.sync()  # 这里要同步各种状态，因为最新状态在 rendezvous。
            if has_set is not None:
                if has_set:
                    msg = (
                        f"The node '{self._node}' has successfully synced its local changes with "
                        f"other nodes in the rendezvous '{self._settings.run_id}'."
                    )
                else:
                    msg = (
                        f"The node '{self._node}' has a stale state and failed to sync its local "
                        f"changes with other nodes in the rendezvous '{self._settings.run_id}'."
                    )

                self._record(message=msg)
                log.debug(msg)

            self._state = self._state_holder.state # 得到最新的状态

            ctx = _RendezvousContext(self._node, self._state, self._settings)

            # Determine the next action to take based on the current state of
            # the rendezvous.
            # 决定下一个操作，state_handler 就是算子
            action = state_handler(ctx, deadline)

            if action == _Action.FINISH:
                continue

            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError()

            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError()

            if action == _Action.SYNC:
                # Delay the execution by one second to avoid overloading the
                # backend if we are asked to poll for state changes.
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:  # 发现当前Action
                    self._add_to_wait_list()  # 然后执行
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()

                # Attempt to sync our changes back to other nodes.
                # 再次同步，把自己状态同步给其他节点
                self._state_holder.mark_dirty()

    '''
    接收到 KEEP_ALIVE action之后，会调用到 _keep_alive 来维持心跳。
    另外，keep_alive 也会在 _add_to_participants等方法内被调用，会更新本地state之中的last heartbeats，
    下一次 sync 时候，会把 last_heartbeats 写入键值存储，这样其他Node就可以知道这个节点的状态了。
    而本地则会在 _sanitize 之中依据 last_heartbeats 做处理，我们之前提到过。
    '''
    def _keep_alive(self) -> None:
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.utcnow()

    #接受到 ADD_TO_PARTICIPANTS 之后，调用 _add_to_participants 从等待列表中移除节点，往 participants 加入这个节点。
    def _add_to_participants(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        state = self._state

        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass

        # The ranks of the participants will be set once the rendezvous is
        # complete.
        state.participants[self._node] = 0

        self._keep_alive()

        if len(state.participants) == self._settings.min_nodes:
            state.deadline = datetime.utcnow() + self._settings.timeout.last_call

        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    # 接受到 ADD_TO_WAIT_LIST 之后，调用 _add_to_wait_list 网 wait_list 中加入节点。
    def _add_to_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        self._state.wait_list.add(self._node)

        self._keep_alive()

    #接受到 REMOVE_FROM_PARTICIPANTS 之后，调用 _remove_from_participants 从 participants 和 last_heartbeats 中删除参与者。
    def _remove_from_participants(self) -> None:
        msg = (
            f"The node '{self._node}' removed itself from the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        state = self._state

        del state.participants[self._node]

        del state.last_heartbeats[self._node]

        # Common epilogue shared with the sanitizer() function of
        # _BackendRendezvousStateHolder.
        _remove_participant_epilogue(state, self._settings)

    # 接受到 REMOVE_FROM_WAIT_LIST 之后，调用 _remove_from_wait_list 从 wait_list 移除节点。
    def _remove_from_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' removed itself from the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        self._state.wait_list.remove(self._node)

        del self._state.last_heartbeats[self._node]

    '''
    state.participants 从哪里来？在 rendezvous 结束时候，会设置 rank。
    因为每个节点上都是按照同样算法排序，所以rank 排序在每个节点上都是一样的。可以保证每个Node得到的rank是与其他Node不同的。
    
    
    接受到 MARK_RENDEZVOUS_COMPLETE 之后，当 rendezvous 聚合操作结束之后，给每一个参与者设置 rank。
每个节点上都是按照同样算法排序，所以rank在每个节点上都是一样的。

    '''
    def _mark_rendezvous_complete(self) -> None:
        msg = (
            f"The node '{self._node}' marked round {self._state.round} of the rendezvous "
            f"'{self._settings.run_id}' as complete. Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)

        state = self._state

        state.complete = True
        state.deadline = None

        # Assign the ranks.
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        msg = (
            f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. "
            "Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)

        self._state.closed = True


def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    """Determines whether a keep-alive heartbeat should be sent."""
    try:
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        return False

    return last_heartbeat <= datetime.utcnow() - ctx.settings.keep_alive_interval

#_RendezvousExitOp 依据当前状态和时间来确定下一步Action。如果本Node不在participants之中，不处理。否则返回一个从 participants 列表删除的下一步Action。如果超时则返回对应Action。
class _RendezvousExitOp:
    """Represents a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.node in ctx.state.participants:
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH

'''
3.2.4 Join
_RendezvousJoinOp 这里依据系统状态不同，做不同处理，比如试图把本Node加入到participant，或者 waiting list，或者继续等待，具体可以参见代码注释。

    从上下文之中提取 _RendezvousState 状态，把结果存放在 state 之中。
    如果状态是closed，则说明此时rendezvous已经结束，则返回_Action.ERROR_CLOSED。
    看看是不是参与者，把结果存放在is_participant。
    如果状态已经结束，且本节点已经是参与者，则说明 rendezvous 可以结束，返回 _Action.FINISH。
    获取当前时间 now。
    如果 now > deadline，说明已经超时。
        如果还有时间做 rollback，说明本节点要返回之前的状态。
        如果本节点已经是参与者，说明此时总节点数目没有达到 min，虽然已经是参与者，但是需要从参与者列表移除，所以返回 _Action.REMOVE_FROM_PARTICIPANTS。
    如果本节点在等待列表之中，说明此时总节点数目没有达到 max，虽然在等待列表之中，但是需要从等待列表移除，所以返回_Action.REMOVE_FROM_WAIT_LIST。
    否则返回_Action.ERROR_TIMEOUT。
    否则没有超时，继续处理。
        如果state.complete 并且本节点不是参与者（如果节点是参与者，前面已经处理过了），说明rendezvous 已经结束，如果还没有达到最大节点数目，并且当前node不在等待列表之中，就需要添加到等待节点列表，等待下次监控周期到的时候，重新做rendezvous，就可以把等待列表中的节点加入到参与列表之中。所以返回_Action.ADD_TO_WAIT_LIST。
        如果本节点是参与者并且state不是complete状态（如果是complete状态，前面已经处理过了），如果已经达到了最小节点数 & 已经超时了，则说明rendezvous 已经结束，则返回_Action.MARK_RENDEZVOUS_COMPLETE。
        否则说明没结束，本节点也不是参与者，则直接加入到参与者列表，返回_Action.ADD_TO_PARTICIPANTS。
    如果需要保持心跳，就返回 _Action.KEEP_ALIVE。
    否则返回_Action.SYNC。
    
    
    
具体逻辑如下：

                           state.closed
                        +-------------------------->   _Action.ERROR_CLOSED
                        |
                        |
                        |  complete & participant
                        +-------------------------->   _Action.FINISH
                        |
                        |
                        |  timeout & participant
                        +-------------------------->   _Action.REMOVE_FROM_PARTICIPANTS
                        |
                        |
                        |  timeout & wait
                        +-------------------------->   _Action.REMOVE_FROM_WAIT_LIST
                        |
+-------------------+   |
|                   |   |  timeout
| _RendezvousJoinOp +------------------------------>   _Action.ERROR_TIMEOUT
|                   |   |
+-------------------+   |  complete & < max & not wait
                        |
                        +-------------------------->   _Action.ADD_TO_WAIT_LIST
                        |
                        |  complete & participant & > min & deadline
                        |
                        +-------------------------->   _Action.MARK_RENDEZVOUS_COMPLETE
                        |
                        |  not complete & not participant
                        |
                        +-------------------------->   _Action.ADD_TO_PARTICIPANTS
                        |
                        |  _should_keep_alive
                        |
                        +-------------------------->   _Action.KEEP_ALIVE
                        |
                        |  else
                        |
                        +-------------------------->   _Action.SYNC

以下是源码之中 ETCD 后端 Rendezvous 状态描述图，我们可以大致参考比对 c10d的状态。



可见，etcd 后端的Join可以分为4个阶段：

setup 阶段，会往固定目录写一个值，这是一个排他锁，如果写失败，说明目前正有一个 rendezvous 过程在进行中。
join（joinable） 阶段。如果写值成功，则进入join 阶段。如果在等待时间结束或者参与训练的节点达到了最大值，则进入 frozen 阶段。
frozen（confirm）阶段。需要所有节点都确认，进入最后的 final 阶段。
final 阶段。分配rank，RANK 0 的实例成为 master。
仿照上图，我们把 c10d 拓展如下。

      +
      |
      |
      v
+-----+------+
|            |
|   closed   +---------------> ERROR_CLOSED
|            |
+-----+------+
      |
      |
      v
+-----+------+  is_participant
|            |
|  complete  +---------------> FINISH
|            |
+-----+------+
      |                                                                                 is_participant
      |
      v                                                                                +----> REMOVE_FROM_PARTICIPANTS
+-----+-------+  now > deadline  +-----------+    now < rollback     +-----------+     |
|             |                  |           |                       |           |     |
|    join     +----------------> |  timeout  +---------------------->+ rollback  +-----+
|             |                  |           |                       |           |     |
+-----+-------+                  +----+------+                       +-----------+     |
      |                               |                                                | in state.wait_list
      |                               |    now > rollback                              |
      |  now < deadline               |                                                +----> REMOVE_FROM_WAIT_LIST
      |                               +---------->  ERROR_TIMEOUT
      |
      |   complete && not is_participant && < max && not in state.wait_list
      |
      +------------------------------------------------------------------>  ADD_TO_WAIT_LIST
      |
      |   not complete && is_participant && > min && > deadline
      |
      +------------------------------------------------------------------>  MARK_RENDEZVOUS_COMPLETE
      |
      |   not complete && not is_participant
      |
      +----------------------------------------->  ADD_TO_PARTICIPANTS
      |
      |   _should_keep_alive
      |
      +--------------------------->  KEEP_ALIVE
      |
      |
      v
     SYNC
    
'''
class _RendezvousJoinOp:
    """Represents a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        state = ctx.state # 从上下文之中提取 _RendezvousState 状态

        # A closed rendezvous means that it no longer accepts new nodes.
        if state.closed:
            return _Action.ERROR_CLOSED # 如果已经结束，就返回 _Action.ERROR_CLOSED

        is_participant = ctx.node in state.participants # 看看是不是参与者

        # If we are part of the rendezvous and it is already complete there is
        # no further action to take.
        if state.complete and is_participant: # 如果是参与者且状态是结束，就返回 _Action.FINISH
            return _Action.FINISH

        now = time.monotonic()
        if now > deadline:  # 如果已经超时
            rollback_period = 5  # 5 seconds

            # If we still have time to rollback (a short period on top of the
            # operation deadline), try to remove ourself from the rendezvous.
            # It is okay if we can't though as our keep-alive will eventually
            # expire.
            if now <= deadline + rollback_period: # 如果还有时间来 rollback
                # If we are part of the rendezvous, it means we couldn't find
                # enough participants to complete it on time.
                if is_participant:   # 此时尚未达到min，虽然已经是参与者，但是需要移除
                    return _Action.REMOVE_FROM_PARTICIPANTS
                # If we are in the wait list, it means we couldn't wait till the
                # next round of the rendezvous.
                if ctx.node in state.wait_list: # 此时已经达到 max，虽然已经在等待列表之中，需要移除
                    return _Action.REMOVE_FROM_WAIT_LIST  # 需要从等待列表移除
            return _Action.ERROR_TIMEOUT  # 返回超时

        if state.complete: # 如果 rendezvous 已经结束
            # If we are here, it means we are not part of the rendezvous. In
            # case the rendezvous has capacity for additional participants add
            # ourself to the wait list for the next round.
            if len(state.participants) < ctx.settings.max_nodes:  # 如果还没有达到最大节点数
                if ctx.node not in state.wait_list:  # 如果当前node不在等待列表之中
                    return _Action.ADD_TO_WAIT_LIST # 就加入到等待列表，发送一个等待action
        elif is_participant:  # 如果已经在参与者列表
            # If the rendezvous has enough number of participants including us,
            # check whether we have passed the rendezvous deadline. If yes,
            # complete it.
            if len(state.participants) >= ctx.settings.min_nodes:   # 如果达到了最小节点数
                if cast(datetime, state.deadline) < datetime.utcnow(): # 如果达到了超时
                    return _Action.MARK_RENDEZVOUS_COMPLETE  # 标示 rendezvous 已经结束
        else:  # 否则就直接加入到参与者
            # The rendezvous is not complete yet and we are not part of it. Try
            # to join.
            return _Action.ADD_TO_PARTICIPANTS

        if _should_keep_alive(ctx):  # 如果需要保持心跳，就返回 _Action.KEEP_ALIVE
            return _Action.KEEP_ALIVE

        # At this point either the rendezvous is not complete, but we are part
        # of it, which means we have to wait for other participants to join; or
        # the rendezvous is complete, but we are not part of it, which means we
        # have to wait for the next round.
        return _Action.SYNC # 否则返回同步状态 _Action.SYNC

#_RendezvousCloseOp 会依据当前状态和时间来确定下一步Action。
class _RendezvousCloseOp:
    """Represents a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.state.closed:
            return _Action.FINISH
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED

'''
3.2 算子
引擎之中实现了一些算子，基本上，一个操作对应一个算子，我们给出几个操作算子的例子，算子就是依据rendezvous的状态来设置操作类型。

3.2.1 心跳
3.2.1.1 检查心跳
_RendezvousKeepAliveOp 的作用是：依据当前状态和时间来确定下一步Action。主要是定期检查本Node是否故障。
'''
class _RendezvousKeepAliveOp:
    """Represents a rendezvous keep-alive update operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if _should_keep_alive(ctx):
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.KEEP_ALIVE
        return _Action.FINISH

'''
DynamicRendezvousHandler 拓展了RendezvousHandler，其定义如下，其最主要逻辑是：在 group_rank = 0 之上建立一个 TCPStore，然后封装成一个 PrefixStore。

最主要的是如下几个成员变量：

    _BackendRendezvousStateHolder 负责在 Rendezvous 之间协调信息。
    _DistributedRendezvousOpExecutor 负责具体执行业务。
    _store 负责保存信息（分布式）
    
我们也可以用如下方式直接生成 DynamicRendezvousHandler。

 store = TCPStore("localhost")
 backend = C10dRendezvousBackend(store, "my_run_id")
 rdzv_handler = DynamicRendezvousHandler.from_backend(
     run_id="my_run_id",
     store=store,
     backend=backend,
     min_nodes=2,
     max_nodes=4
 )
'''
class DynamicRendezvousHandler(RendezvousHandler):
    """Represents a handler that sets up a rendezvous among a set of nodes."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: Optional[_PeriodicTimer]

    '''
    from_backend 是具体生成 DynamicRendezvousHandler 的方法，相当于生成器。

    其生成了 RendezvousSettings，_BackendRendezvousStateHolder 和 node，然后建立了 DynamicRendezvousHandler。
    '''
    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        timeout: Optional[RendezvousTimeout] = None,
    ):
        """Creates a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node adress.
            timeout:
                The timeout configuration of the rendezvous.
        """
        # We associate each handler instance with a unique node descriptor.
        node = cls._node_desc_generator.generate(local_addr)

        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=5),
            keep_alive_max_attempt=3,
        )

        state_holder = _BackendRendezvousStateHolder(backend, settings)

        return cls(node, settings, backend.name, store, state_holder)

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        state_holder: _RendezvousStateHolder,
    ) -> None:
        if not settings.run_id:
            raise ValueError("The run id must be a non-empty string.")

        if settings.min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be greater than zero."
            )

        if settings.max_nodes < settings.min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({settings.max_nodes}) must be greater than or equal "
                f"to the minimum number of nodes ({settings.min_nodes})."
            )

        self._this_node = node

        self._settings = settings

        self._backend_name = backend_name

        self._store = store

        self._state_holder = state_holder

        self._op_executor = _DistributedRendezvousOpExecutor(
            self._this_node, self._state_holder, self._settings
        )

        self._heartbeat_lock = threading.Lock()

        self._keep_alive_timer = None

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: Optional[int] = None,
    ) -> None:
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    @property
    def settings(self) -> RendezvousSettings:
        """Gets the settings of the rendezvous."""
        return self._settings

    def get_backend(self) -> str:
        """See base class."""
        return self._backend_name
    '''
    4.2.2.1 处理成员关系变化
    Elastic 调用 rdzv_handler.next_rendezvous() 来处理成员关系变化，目的是启动下一轮 rendezvous 操作（因为本worker已经启动，需要加入集群）。
    
    注意，next_rendezvous 是 RendezvousHandler 的内部函数。
    这一函数调用会被阻塞，直到 worker 的数量达到了要求。
    在 worker 被初始化，或者重启的时候，这一函数都会被调用。
    当函数返回时，不同的 worker group 会以返回中的 rank 作为唯一的标示。其内部逻辑是：
    
    先使用_RendezvousExitOp让该node退出。
    然后再使用_RendezvousJoinOp把该node重新加入。
    最后启动心跳，返回world size，store等。
    
    
    DynamicRendezvousHandler 之中就体现的不明显，应该是因为 DynamicRendezvousHandler 是在ETCD之后开发，所以很多功能不完善，在演进之中。

本系列是基于PyTorch 1.9 为主进行分析，所以上面 next_rendezvous 代码之中没有错误处理，直接抛到最外面去了。在2021-12月最新代码之中，已经加入了错误处理，后续应该还会继续完善。





4.6 小结
Rendezvous 和 Agent 之间的逻辑联系总结如下，每个启动脚本都有这么一套机制。若干启动脚本的机制之间会互相联系。

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
或者和前面的静态逻辑结合起来看看。

+------------------------+   +----------------------------------------------+     +  +------------------------+   +---------------------------------------------+
| LocalElasticAgent      |   | WorkerSpec                                   |     |  | LocalElasticAgent      |   | WorkerSpec                                  |
|                        |   |                                              |     |  |                        |   |                                             |
| +--------------------+ |   |   rdzv_handler = {DynamicRendezvousHandler} ----+  |  | +--------------------+ |   |  rdzv_handler = {DynamicRendezvousHandler}-----+
| | WorkerGroup        | |   |                                              |  |  |  | | WorkerGroup        | |   |                                             |  |
| |        spec +----------->+   entry = worker_fn                          |  |  |  | |        spec +----------->+  entry = worker_fn                          |  |
| |        workers     | |   |                                              |  |  |  | |        workers     | |   |                                             |  |
| |        store       | |   |   role = {str} 'trainer'                     |  |  |  | |        store       | |   |  role = {str} 'trainer'                     |  |
| |        group_rank  | |   |                                              |  |  |  | |        group_rank  | |   |                                             |  |
| |   group_world_size | |   +----------------------------------------------+  |  |  | |   group_world_size | |   +---------------------------------------------+  |
| |                    | |   +--------------------------------------------+    |  |  | |                    | |   +--------------------------------------------+   |
| +--------------------+ |   | DynamicRendezvousHandler                   |    |  |  | +--------------------+ |   | DynamicRendezvousHandler                   |   |
|  rdzv_run_id           |   |                                            |    |  |  |  rdzv_run_id           |   |                                            |   |
|  store                 |   |                                            |    |  |  |  store                 |   |                                            |   |
+------------------------+   |    _settings: RendezvousSettings           |    |  |  +------------------------+   |    _settings: RendezvousSettings           |   |
                             |                                            | <--+  |                               |                                            +<--+
                             |    _store: Store                           |       |                               |    _store: Store                           |
                             |                                            |       |                               |                                            |
                     +----------+ _state_holder: _RendezvousStateHolder   |       |                               |    _state_holder: _RendezvousStateHolder +-----+
                     |       |                                            |       |                               |                                            |   |
                     |       |    _op_executor: _RendezvousOpExecutor     |       |                               |    _op_executor: _RendezvousOpExecutor     |   |
                     |       |                                            |       |                               |                                            |   |
                     |       +--------------------------------------------+       |                               +--------------------------------------------+   |
                     v                                                            |                                                                                |
           +---------+---------------------+                                      |                                      +-------------------------------+         |
           | _BackendRendezvousStateHolder |                                      |                                      | _BackendRendezvousStateHolder |         |
           |                               |     +-------------------+            |         +--------------------+       |                               |         |
           |             _settings +-----------> | RendezvousSettings|            |         | RendezvousSettings | <----------+ _settings                | <-------+
           |                               |     +-------------------+            |         +--------------------+       |                               |
           |                               |     +-------------------+            |         +--------------------+       |                               |
           |             _state +--------------> | _RendezvousState  |            |         | _RendezvousState   | <----------+ _state                   |
           |                               |     |                   |            |         |                    |       |                               |
           |                               |     +-------------------+            |         +--------------------+       |                               |
           |                               |     +-----------------------+        +         +----------------------+     |                               |
           |             _backend +------------> | C10dRendezvousBackend |                  | C10dRendezvousBackend| <-------+  _backend                 |
           |                               |     |                       |    +---------+   |                      |     |                               |
           |                               |     |             _store +-----> |TCPStore | <-------+ _store         |     |                               |
           |                               |     +---+-------------------+    +---+-----+   +-----------+----------+     |                               |
           |                               |         ^                            |                     ^                |                               |
           |                               |         |                            |                     |                |                               |
           |                               |         |                            |                     |                |                               |
           |             sync +----------------------+                            |                     +---------------------+  sync                    |
           |                               |   set_state                  NODE 1  |  NODE 2               set_state      |                               |
           +-------------------------------+                                      +                                      +-------------------------------+





0x05 总结
目前我们分析了Rendezvous的静态结构和动态逻辑，大家对其机制有了一个基本理解，比如有如下概念：

    节点概念_NodeDesc，这样可以把系统表达出来。
    状态概念。_RendezvousState 是rendezvous的状态。是动态信息，每一个 node 都会维护一个本地 state。
    总体静态类 _BackendRendezvousStateHolder，用来把节点，状态，后端以及其他信息统一维护起来。
    共享共享键值存储，比如TCPStore，可以集中保存上述信息，也可以用来彼此交换信息，达成共识。
    动态server或者handler，RendezvousHandler就提供了一套API以供外界访问。
下一篇我们介绍内部业务逻辑如何实现，即 Rendezvous 引擎。



    '''
    def next_rendezvous(self) -> Tuple[Store, int, int]:
        """See base class."""
        msg = (
            f"The node '{self._this_node}' attempts to join the next round of the rendezvous "
            f"'{self._settings.run_id}'."
        )
        self._record(message=msg)
        log.info(msg)

        try:   # 加入了错误处理
            self._stop_heartbeats()

            # Delay the execution for a small random amount of time if this is our
            # first run. This will slightly skew the rendezvous attempts across the
            # nodes and reduce the load on the backend.
            if self._state_holder.state.round == 0:
                _delay(seconds=(0, 0.3))

            exit_op = _RendezvousExitOp()  # 设置算子
            join_op = _RendezvousJoinOp()  # 设置算子

            deadline = self._get_deadline(self._settings.timeout.join)

            self._op_executor.run(exit_op, deadline) # 这里会进行调用
            self._op_executor.run(join_op, deadline) # 调用

            self._start_heartbeats()

            #这两个变量是动态生成的，所以从 state 之中取出。
            '''
            上面代码之中，使用了 _get_world，这里我们再分析一下。rank, world_size 这两个变量是动态生成的，所以从 state 之中取出。
            而且，因为 participants 是在所有Node之间同步的，所以每个Node得到的 participants 完全一致。
            '''
            rank, world_size = self._get_world()
            store = self._get_store()

        except Exception as e:   # 加入了错误处理，但是没有发起下一轮rendezvous
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

        msg = (
            f"The node '{self._this_node}' has joined round {self._state_holder.state.round} of "
            f"the rendezvous '{self._settings.run_id}' as rank {rank} in a world of size "
            f"{world_size}."
        )
        self._record(message=msg, rank=rank)
        log.info(msg)

        return store, rank, world_size # 返回的是 worker group 的rank

    def is_closed(self) -> bool:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return self._state_holder.state.closed

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def set_closed(self) -> None:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def num_nodes_waiting(self) -> int:
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return len(self._state_holder.state.wait_list)

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def get_run_id(self) -> str:
        """See base class."""
        return self._settings.run_id

    def shutdown(self) -> bool:
        """See base class."""
        self._stop_heartbeats()

        try:
            self._close()

            return True
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to shutdown the rendezvous "
                f"'{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)

            return False
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def _close(self) -> None:
        op = _RendezvousCloseOp() # 设置算子

        deadline = self._get_deadline(self._settings.timeout.close)

        self._op_executor.run(op, deadline)  # 调用

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    #_keep_alive 会调用 _RendezvousKeepAliveOp。
    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        self._heartbeat_lock.acquire()

        op = _RendezvousKeepAliveOp() # 设置算子

        deadline = self._get_deadline(self._settings.timeout.heartbeat)

        try:
            self._op_executor.run(op, deadline) # 调用

            msg = (
                f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous "
                f"'{self._settings.run_id}'."
            )
            self._record(message=msg)
            log.debug(msg)
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the "
                f"rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            log.warning(msg)
        finally:
            self._heartbeat_lock.release()

    '''
    3.2.1.2 定期调用
这里要注意的是，因为做任何算子之前，都要调用 sync 操作，而 sync 会在 node 之间同步状态，因为心跳是定期的，所以同步状态也是定期的。

DynamicRendezvousHandler 之中会启动一个timer，定期调用_keep_alive_weak方法。
    '''
    def _start_heartbeats(self) -> None:
        self._keep_alive_timer = _PeriodicTimer(
            self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self)
        )

        #其次，_keep_alive_weak 会调用 self._keep_alive()。
        self._keep_alive_timer.set_name(f"RendezvousKeepAliveTimer_{self._this_node.local_id}")

        self._keep_alive_timer.start()

    def _stop_heartbeats(self) -> None:
        if self._keep_alive_timer is None:
            return

        self._keep_alive_timer.cancel()

    def _get_world(self) -> Tuple[int, int]:
        state = self._state_holder.state

        return state.participants[self._this_node], len(state.participants)

    def _get_store(self) -> Store:
        key_prefix = f"torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}"

        return PrefixStore(key_prefix, self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        return time.monotonic() + timeout.total_seconds()


def _get_timeout(params: RendezvousParameters, key: str) -> Optional[timedelta]:
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    """Creates a new :py:class:`DynamicRendezvousHandler` from the specified
    parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    """
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
        )

        return DynamicRendezvousHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
