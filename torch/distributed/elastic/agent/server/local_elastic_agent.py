#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import shutil
import signal
import socket
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple

import torch.distributed.elastic.timer as timer
from torch.distributed.elastic import events

from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.events.api import EventMetadataValue
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import PContext, start_processes
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger

log = get_logger()

__all__ = [
    "LocalElasticAgent",
    "TORCHELASTIC_ENABLE_FILE_TIMER",
    "TORCHELASTIC_TIMER_FILE",
]

TORCHELASTIC_ENABLE_FILE_TIMER = "TORCHELASTIC_ENABLE_FILE_TIMER"
TORCHELASTIC_TIMER_FILE = "TORCHELASTIC_TIMER_FILE"

'''
0x05 LocalElasticAgent
LocalElasticAgent 是弹性训练最终使用的代理，主要用于在本地进行操作，负责管理单机上所有的worker进程，其派生了 SimpleElasticAgent。

此代理在每个主机之上部署，并配置为生成n个工作进程。当使用GPU时，n是主机上可用的GPU数量。本地代理不会与部署在其他主机上的其他本地代理通信，即使worker可以在主机间通信。Worker id被解释为本地进程。代理作为把本机所有工作进程作为一个整体启动和停止。

传递给worker的函数和参数必须与python multiprocessing兼容。要将multiprocessing数据结构传递给worker，用户可以在与指定的start_method相同的多处理multiprocessing中创建数据结构，并将其作为函数参数传递。

exit_barrier_timeout用来指定等待其他代理完成的时间量（以秒为单位）。这起到了一个安全网的作用，可以处理worker在不同时间完成的情况，以防止代理将提前完成的worker视为scale-down事件。强烈建议用户代码确保worker以同步方式终止，而不是依赖于exit_barrier_timeout。

SimpleElasticAgent 主要是提供给了初始化和总体运行方式，但是遗留了一些抽象函数没有被实现，比如_start_workers，_stop_workers，_monitor_workers，_shutdown。LocalElasticAgent 就补齐了这些函数。
'''
class LocalElasticAgent(SimpleElasticAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.


    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The ``exit_barrier_timeout`` specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    A named pipe based watchdog can be enabled in ```LocalElasticAgent``` if an
    environment variable ``TORCHELASTIC_ENABLE_FILE_TIMER`` with value 1 has
    been defined in the ```LocalElasticAgent``` process.
    Optionally, another environment variable ```TORCHELASTIC_TIMER_FILE```
    can be set with a unique file name for the named pipe. If the environment
    variable ```TORCHELASTIC_TIMER_FILE``` is not set, ```LocalElasticAgent```
    will internally create a unique file name and set it to the environment
    variable ```TORCHELASTIC_TIMER_FILE```, and this environment variable will
    be propagated to the worker processes to allow them to connect to the same
    named pipe that ```LocalElasticAgent``` uses.

    Example launching function

    ::

        def trainer(args) -> str:
            return "do train"

        def main():  #以下是如何把binary作为入口来启动。  _rendezvous 操作之后，Worker 实例已经生成了，接下来就看看如何生成 Worker 进程。
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint=trainer,
                        args=("foobar",),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            results = agent.run()

            if results.is_failed():
                print("trainer failed")
            else:
                print(f"rank 0 return value: {results.return_values[0]}")
                # prints -> rank 0 return value: do train

    Example launching binary

    ::

        def main():
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        entrypoint="/usr/local/bin/trainer",
                        args=("--trainer-args", "foobar"),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec)
            results = agent.run()

            if not results.is_failed():
                print("binary launches do not have return values")

    """

    def __init__(
        self,
        spec: WorkerSpec,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._pcontext: Optional[PContext] = None
        rdzv_run_id = spec.rdzv_handler.get_run_id()
        self._log_dir = self._make_log_dir(log_dir, rdzv_run_id)
        self._worker_watchdog: Optional[timer.FileTimerServer] = None

    def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
        base_log_dir = log_dir or tempfile.mkdtemp(prefix="torchelastic_")
        os.makedirs(base_log_dir, exist_ok=True)
        dir = tempfile.mkdtemp(prefix=f"{rdzv_run_id}_", dir=base_log_dir)
        log.info(f"log directory set to: {dir}")
        return dir

    def _setup_local_watchdog(self, envs: Dict[int, Dict[str, str]]) -> None:
        enable_watchdog_env_name = TORCHELASTIC_ENABLE_FILE_TIMER
        watchdog_enabled = os.getenv(enable_watchdog_env_name)
        watchdog_file_env_name = TORCHELASTIC_TIMER_FILE
        watchdog_file_path = os.getenv(watchdog_file_env_name)
        if watchdog_enabled is not None and str(watchdog_enabled) == "1":
            if watchdog_file_path is None:
                watchdog_file_path = "/tmp/watchdog_timer_" + str(uuid.uuid4())
            log.info(f"Starting a FileTimerServer with {watchdog_file_path} ...")
            self._worker_watchdog = timer.FileTimerServer(
                file_path=watchdog_file_path,
                max_interval=0.1,
                daemon=True,
                log_event=self._log_watchdog_event)
            self._worker_watchdog.start()
            log.info("FileTimerServer started")
        else:
            log.info(f"Environment variable '{enable_watchdog_env_name}' not found. Do not start FileTimerServer.")
        # Propagate the watchdog file env to worker processes
        if watchdog_file_path is not None:
            for _, worker_env in envs.items():
                worker_env[watchdog_file_env_name] = watchdog_file_path


    def _get_fq_hostname(self) -> str:
        return socket.getfqdn(socket.gethostname())

    def _log_watchdog_event(
        self,
        name: str,
        request: Optional[timer.FileTimerRequest],
    ) -> None:
        wg = self._worker_group
        spec = wg.spec
        md = {
            "watchdog_event": name
        }
        if request is not None:
            md["worker_pid"] = str(request.worker_pid)
            md["scope_id"] = request.scope_id
            md["expiration_time"] = str(request.expiration_time)
            md["signal"] = str(request.signal)
        md_str = json.dumps(md)
        state = "RUNNING"
        metadata: Dict[str, EventMetadataValue] = {
            "run_id": spec.rdzv_handler.get_run_id(),
            "global_rank": None,
            "group_rank": wg.group_rank,
            "worker_id": None,
            "role": spec.role,
            "hostname": self._get_fq_hostname(),
            "state": state,
            "total_run_time": self._total_execution_time,
            "rdzv_backend": spec.rdzv_handler.get_backend(),
            "raw_error": None,
            "metadata": md_str,
            "agent_restarts": spec.max_restarts - self._remaining_restarts,
        }
        # Note: The 'metadata' field of the Event is converted to a TorchelasticStatusLogEntry later.
        #       The 'name' field of the Event is NOT used in the TorchelasticStatusLogEntry.
        event = events.Event(
            name=name, source=events.EventSource.AGENT, metadata=metadata
        )
        events.record(event)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    #以下函数会停止workers。
    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        self._shutdown()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    '''
    _start_workers 方法会调用 start_processes 来启动 worker 进程，默认_start_method 是 "spawn"。
    也就是启动了多个进程，并行执行用户程序。同时这些进程的运行结果会被监控。start_processes 参数之中，entrypoint和args 是用户命令和参数，entrypoint可以是函数或者字符串。

    _start_workers 把 start_processes 方法启动多线程的结果保存在 _pcontext 之中，后续就用 _pcontext 来继续控制，比如结束 worker 就是直接调用 _pcontext 的 close方法。
    '''
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        self._setup_local_watchdog(envs=envs)

        assert spec.entrypoint is not None

        #TE 使用 torch.mp 和 subprocess 包进行多进程处理。在启动多进程时候，把结果保存在 _pcontext 之中，这是一个 PContext 类型的实例。
        ## 把启动多线程的结果保存在 _pcontext 之中。
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,  # 训练代码入口
            args=args,  # 这里重要的是local rank
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )

        return self._pcontext.pids()

    def _shutdown(self, death_sig: signal.Signals = signal.SIGTERM) -> None:
        if self._worker_watchdog is not None:
            self._worker_watchdog.stop()
            self._worker_watchdog = None
        if self._pcontext:
            self._pcontext.close(death_sig)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `torch.distributed.elastic.metrics.prof`.
    #运行之后，TE 会调用 _monitor_workers 对workers进行监控。之前把启动多线程的结果保存在 _pcontext 之中，现在就用 _pcontext 对运行情况进行监控。
    '''
    因为启动和监控涉及到系统整体运行逻辑，需要和 rendezvous 一起才能更好理解，所以我们把这部分的分析推迟，等到 Rendezvous 之后再来做整体分析。

目前总体逻辑如下：

    调用 rdzv_handler.next_rendezvous() 来和其他节点进行同步，获得信息。
    获得信息中的store（可以认为就是远端的KV存储），group_world_size，group_rank 传给 Agent。
    ranks 等信息传给 _assign_worker_ranks方法。
    _assign_worker_ranks 之中，调用 _share_and_gather 在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共KV存储。
    依据 role infos 来确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。
    使用各种信息建立一系列的 Workers。
    Workers 被复制给 Agent 的 WorkerGroup 之中。
    使用 _start_workers 来启动 worker 进程。
    把 worker 进程 id 赋值给 Agent 的 worker.id 之中，这样以后就可以用 worker.id 来操作进程。
    使用 _monitor_workers 监控 worker 进程。
    使用 _exit_barrier 来等待 worker 进程结束。
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
| +--------------------+   +---------+------+  |    7  |               global_rank,      |
|                                    ^         |       |               role_rank,        |
|                                    |         |       |               world_size,       |
|                                    |         |       |               role_world_size)  |
+----------------------------------------------+       |                                 |
                                     |                 +---------------+-----------------+
                                     |                                 |
                                     |                                 | 8
                                     |              9                  v
                                     +-----------------------+   _start_workers
                                                                       +
                                                                       | 10
                                                                       |
                                                                       v
                                                       +---------------+--------------+
                                                       | state = _monitor_workers     |
                                                  +--> |                              +-->
                                                  |    +---------------+--------------+  |
                                                  |                    |                 |
                                                  <--------------------------------------+
                                                     LOOP  Every 30S   |
                                                                       | 11
                                                                       v
                                                                    _exit_barrier
    
_monitor_workers 在监控时候，就使用 _pcontext 进行监控。在监控时候会依据线程结果转为WorkerState.FAILED，WorkerState.HEALTHY 或者WorkerState.SUCCEEDED返回给上层。    


这里会调用 _pcontext.wait(0) 来获取目前 worker 子进程们的状态，然后依据返回结果，转换不同的 WorkerState 返回给调用者。这里就提到了前面讲的，RunResult 应该和 global rank 映射，所以_monitor_workers就有一个从 local rank 到 gloabl rank 的映射。

为何要使用 Global rank 作为进程状态的标示？因为在Node之间需要沟通，这时候需要用Global rank。




3.2 处理
根据返回状态不同，会有不同处理：

如果 WorkerState.SUCCEEDED，则说明训练结束，正常返回。
如果 WorkerState.HEALTHY，则说明训练正常运行，这时候会检查是否有新节点加入，我们后文会详解。
如果 WorkerState.UNHEALTHY, WorkerState.FAILED，说明训练出现问题，这里有两种情况。
一种是程序出错，TE 会进行重试。
一种是节点退出，我们在下文分析，但是其处理流程与程序出错一致。
接下来我们就分析一下如何处理训练结束 和 程序出错。
    '''
    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
        role = worker_group.spec.role
        worker_pids = {w.id for w in worker_group.workers}   # 拿到本agent所有worker的pid
        assert self._pcontext is not None
        pc_pids = set(self._pcontext.pids().values())
        if worker_pids != pc_pids:
            log.error(
                f"[{role}] worker pids do not match process_context pids."
                f" Expected: {worker_pids}, actual: {pc_pids}"
            )
            return RunResult(state=WorkerState.UNKNOWN)

        result = self._pcontext.wait(0) # 对运行结构进行监控
        if result:
            if result.is_failed(): # 如果进程失败
                # map local rank failure to global rank
                worker_failures = {}
                #  返回的结果内部就包括每个进程的运行结果
                for local_rank, failure in result.failures.items():   # local_rank是进程index
                    worker = worker_group.workers[local_rank]   # 拿到对应的worker
                    worker_failures[worker.global_rank] = failure  # 拿到其 global_rank，进而设置worker状态
                return RunResult(
                    state=WorkerState.FAILED,
                    failures=worker_failures,  # 返回运行结果
                )
            else:
                # copy ret_val_queue into a map with a global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,  # 返回运行结果
                )
        else:
            return RunResult(state=WorkerState.HEALTHY)
