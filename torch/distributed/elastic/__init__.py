#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

Torchelastic agent and user worker failover contract:

**TL;DR;**:

* TE(torchelastic) expects user workers to finish with the 5 minutes drift
* It is better to design DDP app to fail for all workers, rather than a single one.
* TE does not synchronize number of restarts between agents
* TE re-rendezvous does not trigger restart decrease
* When a single agent finishes its job(successfully or not), it will close rendezvous.
  If other agents still have workers in progress, they will be terminated.
* Based on above, scale down does not work if at least single agent finishes the job.
* When Scale up is detected by agents, it will not decrease ``max_restarts``


In general TE(torchelastic) can launch arbitrary user code, but there is some
clarifications need to be done around what failover mechanism torchelastic
provides and what failover mechanism it expects from user workers.

Torchelastic currently supports DDP style applications.  That means that
TE expects *ALL* workers finish approximately at the same time. In practice,
it is nearly to impossible to guarantee that all workers in arbitrary
DDP application finish at the time, so TE provides a finalization barrier
that waits for TIMEOUT(5 minutes) for worker finalization.

**Worker Failure**


5.4 其他代理重启
从源码注释可知：新一轮 rendezvous 会让其他 agent 也重启它们的worker。

When worker fails, TE will check the number of restarts available, if there is more than 0 restarts, TE will start a new rendezvous round and restart the worker process. New rendezvous round will other TE agents to terminate their workers.
这是如何做到的？具体如下：

Agent 0（故障Agent）通过 monitoring 发现了故障。
Agent 0 调用 _restart_workers 重启worker。
Agent 0 会调用 next_rendezvous 发起新一轮 rendezvous。
Agent 0 在做任何操作之前，比如 keep alive 操作之前，会调用 sync 来从kvstore获取集群信息，这样可以保证 Agent拿到的是集群最新状态。
Agent 0 会把自己加入到本地的 waiting_list 之中。
Agent 0 同时会调用 mark_dirty，意思是我状态更新了，需要写入KVStore。
Agent 0 会调用sync把自己的waiting_list 被写入 KVStore。
Agent 1（其他正常工作的 agent）会在做任何操作之前，比如 keep alive 操作之前，会调用 sync 操作从KVStore 获取最新信息。
Agent 1 利用这些信息来更新自己的状态，这样本地 waiting_list 就会更新。
Agent 1 的 train loop 在每 30 秒监控之后，因为系统正常，是 Healthy 状态。
Agent 1 所以调用 num_nodes_waiting() 看看 waiting_list 数目。
Agent 1 会获取本地 waiting list 的数目。
如果 waiting list 不为空，也调用_restart_workers。
其最终会调用next_rendezvous。
具体如下：

 Agent 0                                      Agent 1
+---------------------------+                 +--------------------------------------------+
|    _invoke_run            |                 |                       _invoke_run          |
|          +                |                 |                           +                |
|          |                |                 |                           |                |
|          | 1              |                 |                           |                |
|          v                |                 |                           |                |
| Worker Process Error      |                 |                           |                |
|          +                |                 |                           |                |
|          |                |                 |                           | 10             |
|          | 2              |                 |                           v                |
|          v                |                 |                        HEALTHY             |
|  _restart_workers         |                 |                           +                |
|          +                |                 |                           | 11             |
|          |                |                 |                           |                |
|          | 3              |                 |                           v                |
|          v                |                 |              +-->  num_nodes_waiting() > 0 |
|   next_rendezvous         |                 |              |            +                |
|          +                |                 |              |            |                |
|          | 4              |                 |              | 12         | 13             |
|          |                +   +----------+  |              |            v                |
|          v      cluster info  |          |  |              |       _restart_workers      |
|        sync  <------------+-> | KV Store |  |              |            +                |
|          +                |   |          |  |              |            |                |
|          | 5              |   |          |  |              |            | 14             |
|          v                |   |          |  |              |            v                |
|  Add to local waiting_list|   |          |  |              |        next_rendezvous      |
|          +                |   |          |  |              |                             |
|          |                |   |          |  |              |                             |
|          | 6              |   |          |  |              v                             |
|          v                |   |          |  |                                            |
|     mark_dirty            |   |          |  |  Add to local waiting_list                 |
|          +                |   |          |  |              ^                             |
|          |                |   |          |  |              |                             |
|          | 7              |   |          |  |            9 | waiting_list                |
|          v         7      |   |          |  |    8         +                             |
|        sync +---------------> |          +--------------> sync                           |
|              waiting_list |   |          |  |waiting_list                                |
|                           |   +----------+  |                                            |
+---------------------------+                 +--------------------------------------------+


至此，我们监控机制初步介绍完成，因为篇幅所限，我们下一篇继续介绍Scale up/down如何处理。



When worker fails, TE will check the number of restarts
available, if there is more than 0 restarts, TE will start a new rendezvous
round and restart the worker process. New rendezvous round will other
TE agents to terminate their workers.

.. note:: The TE agent does not synchronize restarts between themselves.
          When a single agent performs restart, it will trigger a local ``max_restarts``
          decrease, other agent will not decrease their ``max_restarts``.
          the user to run the distributed application locally on a dev host.

A single worker failure can cause the whole cluster to fail:
If a single worker is constantly failing, it will cause the TE agent
``max_restarts``  to go to zero. This will cause an agent to finish its
work and close rendezvous. If there are any other workers on different
agents, they will be terminated.


**Re-Rendezvous**

Re-rendezvous occurs when TE agents detect a new node
trying to joint a cluster. TE will not decrease ``max_restarts``. TE agents
will terminate its workers and start a new rendezvous round.

Note about DynamicRendezvous(etcd-v2, c10d-experimental): If the rendezvous
has already max_nodes, the new node won't be added to the wait list right
away since there is no need to tear down a rendezvous that is already fully
utilized. The new node will wait until its timeout (600 secs by default)
and periodically check the number of participants. If the number becomes
less than max_nodes, it will be added to the wait list; otherwise, it will time out after 600 secs.

*Scale up event*. When scale up event happens, torchelastic rendezvous
will detect that there are new nodes trying to join. Torchelastic agent
will stop all workers and perform re-rendezvous. Note: when scale up event
happens, *``max_restarts``* will *not* decrease.

*Scale down event*. When scale down event happens, rendezvous will not
notify the torchelastic agent about it. If TE agent launched with ``max_restarts=0`` ,
it relies on the underlying scheduler to handle job restart. If the ``max_restarts>0`` ,
TE agent will terminate workers and start a new rdzv round, which is a *Scale up event*.

"""
