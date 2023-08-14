# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from torch.distributed import Store


class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""


class RendezvousClosedError(RendezvousError):
    """Raised when a rendezvous is closed."""


class RendezvousTimeoutError(RendezvousError):
    """Raised when a rendezvous did not complete on time."""


class RendezvousConnectionError(RendezvousError):
    """Raised when the connection to a rendezvous backend has failed."""


class RendezvousStateError(RendezvousError):
    """Raised when the state of a rendezvous is corrupt."""

'''
RendezvousHandler 用来执行业务逻辑，几个虚函数是：

    next_rendezvous ：rendezvous barrier 的主要入口，新加入的节点会等待在这里，直到当前rendezvous结束，或者超时，或者当前 rendezvous 被标识为closed。
    is_closed ：是否已经结束，如果rendezvous结束，意味着所有试图re-rendezvous都将失败。
    num_nodes_waiting ：返回在rendezvous barrier等待的当前阶段数目，这些节点不属于当前工作组。
    用户应该周期调用这个方法来检查是否有新节点等待加入工作组，如果有，就调用next_rendezvous() (re-rendezvous。) 进行下一次re-rendezvous。
'''
class RendezvousHandler(ABC):
    """Main rendezvous interface.

    Note:
        Distributed Torch users normally **do not** need to implement their own
        ``RendezvousHandler``. An implementation based on C10d Store is already
        provided, and is recommended for most users.
    """

    # 获取 rendezvous backend名字
    @abstractmethod
    def get_backend(self) -> str:
        """Returns the name of the rendezvous backend."""

    # rendezvous barrier 的主要入口，新加入的节点会等待在这里，直到当前rendezvous结束，或者超时，或者当前 rendezvous 被标识为closed。
    @abstractmethod
    def next_rendezvous(
        self,
    ) -> Tuple[Store, int, int]:
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            A tuple of :py:class:`torch.distributed.Store`, ``rank``, and
            ``world size``.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    # 是否已经结束，如果rendezvous结束，意味着所有试图re-rendezvous都将失败
    @abstractmethod
    def is_closed(self) -> bool:
        """Checks whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        """Marks the rendezvous as closed."""

    # 返回在rendezvous barrier等待的当前阶段数目，这些节点不属于当前工作组。用户应该周期调用这个方法来检查是否有新节点等待加入工作组，如果有，就调用`next_rendezvous()` (re-rendezvous。) 进行下一次re-rendezvous。
    @abstractmethod
    def num_nodes_waiting(self) -> int:
        """Returns the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        """Returns the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """

    def shutdown(self) -> bool:
        """Closes all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """

'''
0x03 静态结构
我们接下来看看相关支撑系统。这里要注意的是，elastic 内部有一套 Rendezvous，和 distributed 原有的 Rendezvous 那套不一样，别搞混了。
distributed 原有的 Rendezvous 就是一套简单的 KV 存储。
elastic Rendezvous 则要复杂得多。

我们仔细分析一下 Rendezvous 的支撑系统。

3.1 启动参数
RendezvousParameters 是构建RendezvousHandler所需参数。

    backend ：后端名称。
    endpoint ：端点，格式是 [:]。
    run_id : rendezvous 的 id。
    min_nodes ：rendezvous 的最小节点数目。
    max_nodes ：rendezvous 的最大节点数目。
    kwargs ：后端的附加参数。
'''
class RendezvousParameters:
    """Holds the parameters to construct a :py:class:`RendezvousHandler`.

    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        **kwargs,
    ):
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs
        self.local_addr = local_addr

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the value for ``key`` if ``key`` exists, else ``default``."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Returns the value for ``key`` as a ``bool``."""
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Returns the value for ``key`` as an ``int``."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e


RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]

'''
在 torch/distributed/elastic/rendezvous/api.py 之中有如下代码。

# The default global registry instance used by launcher scripts to instantiate
# rendezvous handlers.
rendezvous_handler_registry = RendezvousHandlerRegistry()
所以我们来到了 RendezvousHandlerRegistry。

4.3.1 RendezvousHandlerRegistry
RendezvousHandlerRegistry 是一个负责创建 RendezvousHandler 的工厂类。

    register 就是往内部字典添加对应的构建器。
    create_handler 就是依据key，取出对应的构建器。
    rendezvous_handler_registry 是全局Registry。
'''
class RendezvousHandlerRegistry:
    """Represents a registry of :py:class:`RendezvousHandler` backends."""

    _registry: Dict[str, RendezvousHandlerCreator]

    def __init__(self) -> None:
        self._registry = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        """Registers a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        current_creator: Optional[RendezvousHandlerCreator]
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None

        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Creates a new :py:class:`RendezvousHandler`."""
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call `{self.register.__name__}`?"
            ) from e

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )

        return handler


# The default global registry instance used by launcher scripts to instantiate
# rendezvous handlers.

rendezvous_handler_registry = RendezvousHandlerRegistry() #系统会创建一个全局的 registry
'''
在这里注册了若干handler，用来提供创建器。
rendezvous 提供了如下实现，分别是 etcd、etcd-v2、c10d 和 static。

from .api import rendezvous_handler_registry as handler_registry

def _register_default_handlers() -> None:
    handler_registry.register("etcd", _create_etcd_handler)
    handler_registry.register("etcd-v2", _create_etcd_v2_handler)
    handler_registry.register("c10d", _create_c10d_handler)
    handler_registry.register("static", _create_static_handler)
运行时候就是：

rendezvous_handler_registry = 
  _registry = {dict: 4} 
   'etcd' = {function} <function _create_etcd_handler at 0x7ff657e12d08>
   'etcd-v2' = {function} <function _create_etcd_v2_handler at 0x7ff657e12d90>
   'c10d' = {function} <function _create_c10d_handler at 0x7ff657e12e18>
   'static' = {function} <function _create_static_handler at 0x7ff657b9d2f0>
   __len__ = {int} 4

其含义就是：_create_etcd_handler 可以创建 etcd 类型的handler，以此类推。
'''
