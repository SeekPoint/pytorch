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

    def sync(self) -> Optional[bool]:
        """See base class."""
        state_bits: Optional[bytes] = None

        token = None

        has_set: Optional[bool]

        if self._dirty:
            has_set = False

            state_bits = pickle.dumps(self._state)

            # 这里会对后端进行设置
            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:
            has_set = None

            if self._cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time >= max(time.monotonic() - self._cache_duration, 0):
                    return None

            get_response = self._backend.get_state()
            if get_response is not None:
                state_bits, token = get_response

        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
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
            del state.last_heartbeats[dead_node]

            try:
                del state.participants[dead_node]

                participant_removed = True
            except KeyError:
                pass

            try:
                state.wait_list.remove(dead_node)
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



    '''

    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
    ) -> None:
        """See base class."""
        action = None

        while action != _Action.FINISH:
            # Reads or writes the latest rendezvous state shared by all nodes in
            # the rendezvous. Note that our local changes might get overridden
            # by another node if that node synced its changes before us.
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

    def _keep_alive(self) -> None:
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.utcnow()

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

    def _add_to_wait_list(self) -> None:
        msg = (
            f"The node '{self._node}' added itself to the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        log.debug(msg)

        self._state.wait_list.add(self._node)

        self._keep_alive()

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


class _RendezvousExitOp:
    """Represents a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.node in ctx.state.participants:
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH


class _RendezvousJoinOp:
    """Represents a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        state = ctx.state

        # A closed rendezvous means that it no longer accepts new nodes.
        if state.closed:
            return _Action.ERROR_CLOSED

        is_participant = ctx.node in state.participants

        # If we are part of the rendezvous and it is already complete there is
        # no further action to take.
        if state.complete and is_participant:
            return _Action.FINISH

        now = time.monotonic()
        if now > deadline:
            rollback_period = 5  # 5 seconds

            # If we still have time to rollback (a short period on top of the
            # operation deadline), try to remove ourself from the rendezvous.
            # It is okay if we can't though as our keep-alive will eventually
            # expire.
            if now <= deadline + rollback_period:
                # If we are part of the rendezvous, it means we couldn't find
                # enough participants to complete it on time.
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                # If we are in the wait list, it means we couldn't wait till the
                # next round of the rendezvous.
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT

        if state.complete:
            # If we are here, it means we are not part of the rendezvous. In
            # case the rendezvous has capacity for additional participants add
            # ourself to the wait list for the next round.
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
        elif is_participant:
            # If the rendezvous has enough number of participants including us,
            # check whether we have passed the rendezvous deadline. If yes,
            # complete it.
            if len(state.participants) >= ctx.settings.min_nodes:
                if cast(datetime, state.deadline) < datetime.utcnow():
                    return _Action.MARK_RENDEZVOUS_COMPLETE
        else:
            # The rendezvous is not complete yet and we are not part of it. Try
            # to join.
            return _Action.ADD_TO_PARTICIPANTS

        if _should_keep_alive(ctx):
            return _Action.KEEP_ALIVE

        # At this point either the rendezvous is not complete, but we are part
        # of it, which means we have to wait for other participants to join; or
        # the rendezvous is complete, but we are not part of it, which means we
        # have to wait for the next round.
        return _Action.SYNC


class _RendezvousCloseOp:
    """Represents a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        if ctx.state.closed:
            return _Action.FINISH
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED


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

            exit_op = _RendezvousExitOp()
            join_op = _RendezvousJoinOp()

            deadline = self._get_deadline(self._settings.timeout.join)

            self._op_executor.run(exit_op, deadline)
            self._op_executor.run(join_op, deadline)

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
        op = _RendezvousCloseOp()

        deadline = self._get_deadline(self._settings.timeout.close)

        self._op_executor.run(op, deadline)

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        self._heartbeat_lock.acquire()

        op = _RendezvousKeepAliveOp()

        deadline = self._get_deadline(self._settings.timeout.heartbeat)

        try:
            self._op_executor.run(op, deadline)

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

    def _start_heartbeats(self) -> None:
        self._keep_alive_timer = _PeriodicTimer(
            self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self)
        )

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
