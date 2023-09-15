# -*- coding: utf-8 -*-
# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""Implements "Block Partitions of Sequences" by Imre Bárány et al.

Paper: https://arxiv.org/pdf/1308.2452.pdf

"""
from typing import Iterator, List, Tuple

__all__ = ["solve"]

'''
1.5 分割算法
得到每层的计算时间或者内存大小之后，会通过如下代码来进行具体分割。
    times = profile_times(module, sample, timeout, torch.device(device))
    return balance_cost(times, partitions)
    
具体 balance_cost 只是一个封装而已，算法还是 blockpartition.solve。
    def balance_cost(cost: List[int], partitions: int) -> List[int]:
        partitioned = blockpartition.solve(cost, partitions)
        return [len(p) for p in partitioned]
        
从其注释可知，blockpartition.solve 实现了这篇论文的算法。
Implements "Block Partitions of Sequences" by Imre Bárány et al.Paper: https://arxiv.org/pdf/1308.2452.pdf
这是一篇数学论文，其算法伪代码如下（与后续实现中注释基本一一对应）。
    01-04.png
该论文是纯粹的数学论证，我们不去研究其内部机制，只是看看其运行结果。

我们回忆一下，这里支持的模型是顺序模型，所以无论时间还是内存大小，都是一个list。solve的作用就是把这个list尽量平均分配成若干组。
假设模型有6层，每层的运行时间如下，需要分配到两个device之上，那么应该如何分割呢？
    blockpartition.solve([1, 2, 3, 4, 5, 6], partitions=2) # 就是第一层运行时间是1个单位，第二层运行时间是2个单位，依次类推。
    结果是 [[1, 2, 3, 4], [5, 6]]，可以看到，这个6个层被比较均匀的按照运行时间分成了两个partition
如果分成三个device，则：
    solve([1, 2, 3, 4, 5, 6], partitions=3)
    结果是 [[1, 2, 3], [4, 5], [6]]，可以看到，这个6个层被比较均匀的按照运行时间分成了三个partition
然后 balance_cost 会获得每一个 partition 的具体层数，得到balance的最终是：
    [3,2,1]
分区算法具体代码如下，有兴趣的朋友可以结合论文仔细研究。
'''
def solve(sequence: List[int], partitions: int = 1) -> List[List[int]]:
    """Splits a sequence into several partitions to minimize variance for each
    partition.

    The result might not be optimal. However, it can be done only in O(kn³),
    where k is the number of partitions and n is the length of the sequence.

    """
    if partitions < 1:
        raise ValueError(f"partitions must be a positive integer ({partitions} < 1)")

    n = len(sequence)
    if n < partitions:
        raise ValueError(f"sequence is shorter than intended partitions ({n} < {partitions})")

    # Normalize the sequence in [0, 1].
    minimum = min(sequence)
    maximum = max(sequence) - minimum

    normal_sequence: List[float]
    if maximum == 0:
        normal_sequence = [0 for _ in sequence]
    else:
        normal_sequence = [(x - minimum) / maximum for x in sequence]

    splits = [n // partitions * (x + 1) for x in range(partitions - 1)] + [n]

    def block_size(i: int) -> float:
        start = splits[i - 1] if i > 0 else 0
        stop = splits[i]
        return sum(normal_sequence[start:stop])

    def leaderboard() -> Iterator[Tuple[float, int]]:
        return ((block_size(i), i) for i in range(partitions))

    while True:
        """
        (1) Fix p ∈ [k] with M(P) = bp. So Bp is a maximal block of P.
        """
        # max_size: M(P)
        max_size, p = max(leaderboard())

        while True:
            """
            (2) If M(P) ≤ m(P) + 1, then stop.
            """
            # min_size: m(P)
            min_size, q = min(leaderboard())

            if max_size <= min_size + 1:
                return [sequence[i:j] for i, j in zip([0] + splits[:-1], splits)]

            """
            (3) If M(P) > m(P) + 1, then let m(P) = bq for the q ∈ [k] which is
            closest to p (ties broken arbitrarily). Thus Bq is a minimal block
            of P. Let Bh be the block next to Bq between Bp and Bq. (Note that
            Bh is a non-empty block: if it were, then m(P) = 0 and we should
            have chosen Bh instead of Bq.)
            """
            if p < q:
                """
                So either p < q and then h = q−1 and we define P ∗ by moving
                the last element from Bh = Bq−1 to Bq,
                """
                h = q - 1
                splits[h] -= 1
            else:
                """
                or q < p, and then h = q + 1 and P ∗ is obtained by moving the
                first element of Bh = Bq+1 to Bq.
                """
                h = q + 1
                splits[q] += 1

            """
            Set P = P ∗ . If p = h, then go to (1), else go to (2).
            """
            if p == h:
                break
