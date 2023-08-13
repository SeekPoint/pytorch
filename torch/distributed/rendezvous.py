try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional

from torch.distributed import FileStore, PrefixStore, Store, TCPStore

from .constants import default_pg_timeout

'''
0x03 Store
我们给出一个正式的概念。Store 是分布式包（distributed package）所提供的分布式键值存储，所有的 workers 都会访问这个存储以共享信息以及初始化分布式包 。用户可以通过显式创建存储来作为init_method的替代。目前有 3 种键值存储：TCPStore， FileStore，和HashStore。

我们接着上节继续看 handler 概念。

3.1 _rendezvous_handlers
在 PyTorch 定义了一个全局变量 _rendezvous_handlers，用来保存如何返回 store 的方法，可以认为是工厂方法。
'''
_rendezvous_handlers = {}

#注册代码如下，就是往全局变量之中插入handler。
def register_rendezvous_handler(scheme, handler):
    """Registers a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Args:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(
            "Rendezvous handler for {}:// already registered".format(scheme)
        )
    _rendezvous_handlers[scheme] = handler


# Query will have format "rank=0&world_size=1" and is
# converted into {"rank": 0, "world_size": 1}
def _query_to_dict(query: str) -> Dict[str, str]:
    return {pair[0]: pair[1] for pair in (pair.split("=") for pair in filter(None, query.split("&")))}


def _rendezvous_helper(url: str, rank: int, world_size_opt: Optional[int], **kwargs):
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == "env":
            rank = int(os.environ.get("RANK", rank))
            # If the world_size env variable is not present then it is a dynamic group
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        assert (
            "rank" not in query_dict and "world_size" not in query_dict
        ), "The url: {url} has node-specific arguments(rank, world_size) already.".format(
            url=url
        )
        if rank != -1:
            query_dict["rank"] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict["world_size"] = str(world_size)
        result = result._replace(
            query="{}".format(
                "&".join(["{}={}".format(k, v) for k, v in query_dict.items()])
            )
        )
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError("No rendezvous handler for {}://".format(result.scheme))
    return _rendezvous_handlers[result.scheme](url, **kwargs)

'''
2.3 rendezvous
上面代码之中提到了 rendezvous，我们就来看看这个概念。

在我们可以运行集合算法之前，参与的进程需要找到彼此并交换信息才能够进行通信。我们称这个过程为rendezvous。rendezvous过程的结果是一个三元组，其中包含一个共享键/值存储（store），进程的等级（rank）和参与进程的总数。如果内置的rendezvous方法都不适用于您的执行环境，那么您可以选择注册自己的rendezvous处理程序。在调用rendezvous函数时，选择一个唯一的名称并使用URL方案来标识它。

rendezvous 方法就是依据参数，选择不同的handler来处理。


rendezvous 具体就是依据 init_method 来选择一个 _rendezvous_handler，然后 _rendezvous_handler 返回了 store。
'''
def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    if not isinstance(url, str):
        raise RuntimeError("`url` must be a string. {}: {}".format(type(url), url))

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError("`rank` must be an integer. {}".format(rank))

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError("`world_size` must be an integer. {}".format(world_size))

    return _rendezvous_helper(url, rank, world_size, **kwargs)


def _create_store_from_options(backend_options, rank):
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)

'''
3.2 handlers
如果仔细看 handlers 的代码，就会发现其就是返回了不同的 store，比如 _tcp_rendezvous_handler具体就是使用各种信息建立 TCPStore，然后返回。

以下代码均删除非关键代码。

3.2.1 _file_rendezvous_handler
这里返回了FileStore。
'''
def _file_rendezvous_handler(url: str, **kwargs):
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if sys.platform == "win32":
        import urllib.request

        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # Normalizing an empty string produces ".", which is not expected.
            path = os.path.normpath(path)

    if not path:
        raise _error("path missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _torchelastic_use_agent_store() -> bool:
    return os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None) == str(True)


def _create_c10d_store(hostname, port, rank, world_size, timeout) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether
    we need to re-use agent store. The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that
    the agent leader (node rank 0) hosts the TCPStore server (for which the
    endpoint is specified by the given ``hostname:port``). Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host
    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname
    and port are correctly passed via ``hostname`` and ``port``. All
    non-zero ranks will create and return a TCPStore client.
    """
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    if _torchelastic_use_agent_store():
        attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
    else:
        start_daemon = rank == 0
        return TCPStore(
            hostname, port, world_size, start_daemon, timeout, multi_tenant=True
        )

#这里返回了 TCPStore。
def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    assert result.hostname is not None

    store = _create_c10d_store(result.hostname, result.port, rank, world_size, timeout)

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using tcp:// method")

#居然也返回了 TCPStore，但是其会从环境变量中提取需要的信息。
def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        return _error("environment variable %s expected, but not set" % var)

    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    result = urlparse(url)
    query_dict = _query_to_dict(result.query)

    rank: int
    world_size: int
    master_port: int
    master_addr: str

    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))

    store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout)

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")

#handler 如下，你会发现，其实 handler 就是对应了初始化的三种方法：
register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)

'''
2.4 小结
从目前分析结果来看，我们得到了如下结论：

init_method 最终还是落到了 store 之上，store才是起作用的实体。
参与的进程需要找到彼此并交换信息才能够进行通信。这个过程被称为rendezvous。
'''
