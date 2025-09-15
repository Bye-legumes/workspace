# Problem — Minimize Padding Cost with at most `Bmax` Batches

## Problem Statement
You are given an array `L[1..N]` of **nondecreasing** positive integers (token lengths). You must partition the array into at most `Bmax` **contiguous** groups (keeping order).  
For any group `L[j..i]`, define its padding cost as:
\[
\text{cost}(j,i) = (i-j+1)\cdot \max(L[j..i]) - \sum_{k=j}^{i} L[k]
\]
Because `L` is nondecreasing, \(\max(L[j..i]) = L[i]\).

**Goal:** minimize the total padding cost across all groups, subject to using at most `Bmax` groups.

**Input:** `N`, `Bmax`, and a nondecreasing array `L[1..N]`  
**Output:** the minimal total cost; optionally, one optimal partition.

---

## Key Ideas

### 1) Dynamic Programming (Exact Optimal)
Let `S[i] = L[1]+...+L[i]` (prefix sums, with `S[0]=0`).

Define `dp[b][i]` = minimal total cost to cover the **first `i` items** using **exactly `b` groups**.

Transition (last group is `j..i`):
\[
dp[b][i] = \min_{1\le j\le i}\Big( dp[b-1][j-1] + (i-j+1)\cdot L[i] - (S[i]-S[j-1]) \Big)
\]

Initialization: `dp[0][0] = 0`, all other states are `+∞`.

Answer: \(\min_{1\le b\le B_{max}} dp[b][N]\).  
(If you require “**at most** `Bmax`”, take the minimum over `b=1..Bmax`.)

**Complexity:** `O(Bmax * N^2)` time, `O(Bmax * N)` space.  
This is usually fine for medium constraints. With additional structure/assumptions, divide-and-conquer optimization may apply; the plain `O(BN^2)` version is the safest baseline.

### 2) Cost Computation in O(1)
Because `L` is nondecreasing, for any interval `j..i`:
- `max = L[i]`
- `sum = S[i] - S[j-1]`
- `cost(j,i) = (i-j+1) * L[i] - (S[i] - S[j-1])`

So each transition is O(1), and the `N^2` comes from enumerating `j`.

### 3) Reconstructing an Optimal Partition
Track the `argmin j` used for each `dp[b][i]`. After finishing DP, pick the best `b*` (≤ `Bmax`) for `i=N` and backtrack to recover the groups.

---

## Python Solution (DP with Reconstruction)

```python
from math import inf
from itertools import accumulate
from typing import List, Tuple

def minimize_padding_with_bmax_batches(lengths: List[int], Bmax: int, assume_sorted: bool = True
                                      ) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Minimize total padding cost when partitioning nondecreasing 'lengths' into at most Bmax contiguous batches.

    Returns:
        total_cost: int
        partitions: list of (start_index, end_index) in 1-based indices, covering [1..N] in order.
    """
    if not lengths:
        return 0, []

    # If needed, sort (Problem A assumes nondecreasing input)
    L = lengths if assume_sorted else sorted(lengths)
    n = len(L)

    # Prefix sums S[i] = sum of first i elements, S[0] = 0
    S = [0] + list(accumulate(L))

    # dp[b][i] = min cost to cover first i items with exactly b batches
    dp = [[inf] * (n + 1) for _ in range(Bmax + 1)]
    prev = [[-1]  * (n + 1) for _ in range(Bmax + 1)]  # prev[b][i] = starting index j of last batch
    dp[0][0] = 0

    # Transition
    for b in range(1, Bmax + 1):
        for i in range(1, n + 1):
            # last batch is j..i
            max_len = L[i - 1]  # since nondecreasing, max in j..i is L[i-1]
            best_val = inf
            best_j = -1
            # Enumerate start j of the last batch
            for j in range(1, i + 1):
                # cost(j,i) = (i-j+1)*max - sum(L[j..i])
                batch_size = i - j + 1
                batch_sum = S[i] - S[j - 1]
                cost = batch_size * max_len - batch_sum
                cand = dp[b - 1][j - 1] + cost
                if cand < best_val:
                    best_val = cand
                    best_j = j
            dp[b][i] = best_val
            prev[b][i] = best_j

    # Choose best number of batches <= Bmax
    best_cost = inf
    best_b = -1
    for b in range(1, Bmax + 1):
        if dp[b][n] < best_cost:
            best_cost = dp[b][n]
            best_b = b

    # Reconstruct partitions
    partitions = []
    i = n
    b = best_b
    while b > 0 and i > 0:
        j = prev[b][i]
        partitions.append((j, i))  # 1-based indices
        i = j - 1
        b -= 1
    partitions.reverse()

    return int(best_cost), partitions

# ---- Example usage (expected behavior) ----
if __name__ == "__main__":
    L = [1, 2, 3, 3, 3, 6, 10, 19]  # already nondecreasing
    Bmax = 2
    cost, parts = minimize_padding_with_bmax_batches(L, Bmax)
    print("Min cost:", cost)        # Expect 25
    print("Partitions:", parts)     # Expect [(1,5), (6,8)] i.e., [1,2,3,3,3] | [6,10,19]
```

# Sub-Optimal but Faster via `np.histogram` Bucketing + Greedy Merging

## Motivation
The exact DP for **Problem A** is optimal but costs `O(Bmax * N^2)`.  
When `N` is large (millions) you may prefer a **histogram-based approximation**:

1) **Bucket** lengths into adjacent ranges (bins) so items in a bin have similar lengths.  
2) **Greedily merge** adjacent bins to hit a target number of batches while adding the **least extra padding** each time.

This keeps batches “length-homogeneous” and reduces padding with **linear or near-linear** passes, avoiding quadratic DP.

---

## Idea

### Per-batch padding cost
For a (sub)batch with:
- `c` items,
- length sum `s`,
- maximum length `m`,

its padding cost is:
\[
\text{cost} = c \cdot m - s
\]

### Greedy merge rule
Given two **adjacent** groups `A` and `B` with stats `(cA, sA, mA)` and `(cB, sB, mB)`, merging them yields:
- `c = cA + cB`
- `s = sA + sB`
- `m = max(mA, mB)`

Extra padding introduced by this merge:
\[
\Delta = \big((cA + cB)\cdot m - (sA + sB)\big) - \big(cA\cdot mA - sA + cB\cdot mB - sB\big)
\]
Greedily pick the adjacent pair with **minimum** `Δ` and merge, repeat until the number of groups ≤ `Bmax`.

> Why adjacent only? We keep bins ordered by length so merged groups remain contiguous in length space (consistent with the original problem’s “contiguous” notion after sorting).

---

## Algorithm

1. **Sort** lengths `L` ascending (if not already).
2. **Bucketization** (choose one):
   - **Equal-width**: `np.histogram(L, bins=K)`  
   - **Quantile bins** (recommended): boundaries at quantiles to keep similar counts per bin; more stable with long tails.
3. For each **non-empty** bin, compute:
   - `count c`, `sum s`, and `max m` (max over the elements assigned to this bin).
4. Build a **min-heap** of `Δ` for all **adjacent** bin pairs.
5. While `#bins > Bmax`: pop the smallest `Δ`, merge that pair, and **update only its neighbors’ Δ**.
6. (Optional) If you also have a **max_tokens** cap `T` per batch (i.e., `(#items) * max_len ≤ T`), reject merges that would violate it (or pre-split oversized bins).
7. After merging stops, your final bins are the batches. The total cost is the sum of `c*m - s` over bins.

**Complexity (typical):**
- Bucketization: `O(N)` (quantile edges need a partial sort or selection; NumPy’s `quantile` is near-linear in practice).
- Computing stats: `O(N)`.
- Merging: at most `(initial_bins - Bmax)` merges, each `O(log initial_bins)` due to the heap.  
  Overall ≪ quadratic.

---

## Trade-offs

- **Pros:** Very fast, simple, scalable; good in practice when lengths are roughly unimodal or have smooth tails.
- **Cons:** Sub-optimal vs. DP; quality depends on the **initial number of bins** and **binning scheme**.  
- **Tips:** Start with `K ≈ 50–200` bins; use **quantile bins** for heavy-tailed data.

---

## Python Implementation

```python
import numpy as np
import heapq
from typing import List, Literal, Tuple, Optional

def histogram_greedy_batching(
    lengths: List[int],
    Bmax: int,
    init_bins: int = 100,
    binning: Literal["equalwidth", "quantile"] = "quantile",
    max_tokens: Optional[int] = None,
    assume_sorted: bool = True,
):
    """
    Sub-optimal but fast: bucket lengths via histogram, then greedily merge adjacent bins
    minimizing the incremental padding cost Δ until number of bins <= Bmax.

    Args:
        lengths: list of positive ints (token lengths).
        Bmax: target maximum number of batches after merging.
        init_bins: initial number of histogram bins before merging.
        binning: "quantile" (recommended) or "equalwidth".
        max_tokens: optional per-batch cap: (#items) * max_len <= max_tokens.
        assume_sorted: if False, will sort ascending.

    Returns:
        total_cost: int
        batches: list of dicts, each with keys:
                 {"count", "sum", "max", "left_idx", "right_idx", "left_edge", "right_edge"}
                 left/right_idx index the sorted array region covered by this batch.
                 left/right_edge are the bin edges in length space (for reference).
    """
    if len(lengths) == 0:
        return 0, []

    # 1) Sort ascending to align with "contiguous by length" idea
    L = np.asarray(lengths, dtype=np.int64)
    if not assume_sorted:
        L = np.sort(L)
    n = L.shape[0]

    # Safety: Bmax cannot exceed n (point partitions).
    Bmax = max(1, min(Bmax, n))

    # 2) Build bin edges
    if binning == "equalwidth":
        lo, hi = int(L[0]), int(L[-1])
        if lo == hi:
            # All equal: single bin
            edges = np.array([lo, hi + 1], dtype=float)
        else:
            edges = np.linspace(lo, hi, init_bins + 1, dtype=float)
    elif binning == "quantile":
        # Quantile edges; ensure strictly increasing with unique values
        qs = np.linspace(0, 1, init_bins + 1)
        edges = np.quantile(L, qs)
        # De-duplicate edges to avoid empty/degenerate bins due to ties
        edges = np.unique(edges)
        if edges.size == 1:
            edges = np.array([edges[0], edges[0] + 1.0], dtype=float)
    else:
        raise ValueError("binning must be 'equalwidth' or 'quantile'")

    # 3) Assign items to bins
    # digitize uses right-exclusive by default with bins as edges
    bin_idx = np.clip(np.digitize(L, edges[1:-1], right=False), 0, edges.size - 2)

    # Prepare per-bin stats by grouping indices
    # We'll materialize segments (contiguous in the sorted array) for each bin
    bins = []
    start = 0
    for b in range(edges.size - 1):
        # Find contiguous slice for bin b
        # Since bin_idx is nondecreasing (L is sorted), we can scan
        while start < n and bin_idx[start] < b:
            start += 1
        end = start
        while end < n and bin_idx[end] == b:
            end += 1
        if start == end:
            continue  # empty bin, skip

        seg = L[start:end]
        c = int(seg.size)
        s = int(seg.sum())
        m = int(seg[-1])  # max because seg is sorted
        batch = {
            "count": c,
            "sum": s,
            "max": m,
            "left_idx": start,
            "right_idx": end - 1,
            "left_edge": float(edges[b]),
            "right_edge": float(edges[b + 1]),
            "alive": True,
            "left": None,   # pointers for neighbor maintenance
            "right": None,
            "id": b,        # unique id for heap invalidation
            "rev": 0,       # local revision counter
        }
        bins.append(batch)
        start = end

    # If all items collapsed into fewer than Bmax bins, we're already done
    if len(bins) == 0:
        # Should not happen (n>0), but guard anyway
        return 0, []
    if len(bins) <= Bmax:
        total_cost = sum(b["count"] * b["max"] - b["sum"] for b in bins)
        return int(total_cost), bins

    # Link neighbors (doubly-linked list by indices in 'bins')
    for i in range(len(bins)):
        bins[i]["left"] = i - 1 if i > 0 else None
        bins[i]["right"] = i + 1 if i < len(bins) - 1 else None

    # Helper: compute cost of a bin & Δ for merging two neighbors
    def bin_cost(b):
        return b["count"] * b["max"] - b["sum"]

    def merge_delta(i_left, i_right):
        A = bins[i_left]
        B = bins[i_right]
        if (not A["alive"]) or (not B["alive"]):
            return None
        # Respect max_tokens if provided: merging must satisfy (#items)*max_len <= T
        merged_count = A["count"] + B["count"]
        merged_max = max(A["max"], B["max"])
        if (max_tokens is not None) and (merged_count * merged_max > max_tokens):
            return None  # illegal merge
        merged_sum = A["sum"] + B["sum"]
        cost_merge = merged_count * merged_max - merged_sum
        delta = cost_merge - (bin_cost(A) + bin_cost(B))
        return delta

    # Build initial heap of candidate merges among adjacent pairs
    # We also store a snapshot of (rev_left, rev_right) for invalidation
    heap = []
    def push_pair(i_left, i_right):
        if i_left is None or i_right is None:
            return
        d = merge_delta(i_left, i_right)
        if d is None:
            return
        entry = (d, i_left, i_right, bins[i_left]["rev"], bins[i_right]["rev"])
        heapq.heappush(heap, entry)

    for i in range(len(bins) - 1):
        push_pair(i, i + 1)

    alive_count = len(bins)

    # Greedy merge until enough bins
    while alive_count > Bmax and heap:
        d, iL, iR, revL, revR = heapq.heappop(heap)

        # Validate entry still current
        if not (0 <= iL < len(bins) and 0 <= iR < len(bins)):
            continue
        A, B = bins[iL], bins[iR]
        if (not A["alive"]) or (not B["alive"]):
            continue
        if A["rev"] != revL or B["rev"] != revR:
            continue  # out-of-date heap entry; skip

        # Re-check current Δ (since stats may have changed via other merges)
        current_delta = merge_delta(iL, iR)
        if current_delta is None:
            continue
        # We proceed with the merge
        # Create merged stats in A; mark B dead
        A["count"] += B["count"]
        A["sum"] += B["sum"]
        A["max"] = max(A["max"], B["max"])
        A["right_idx"] = max(A["right_idx"], B["right_idx"])
        A["right_edge"] = max(A["right_edge"], B["right_edge"])
        A["rev"] += 1  # bump revision due to mutation

        # Stitch neighbors: A absorbs B
        right_of_B = B["right"]
        A["right"] = right_of_B
        if right_of_B is not None:
            bins[right_of_B]["left"] = iL

        B["alive"] = False
        alive_count -= 1

        # Push new candidate merges involving A and its neighbors
        push_pair(A["left"], iL)     # left neighbor with A
        push_pair(iL, A["right"])    # A with right neighbor

    # Compute final total cost & collect alive bins as batches
    batches = [b for b in bins if b["alive"]]
    total_cost = sum(bin_cost(b) for b in batches)
    return int(total_cost), batches


# ------------------ Demo ------------------
if __name__ == "__main__":
    L = [1, 2, 3, 3, 3, 6, 10, 19]
    Bmax = 2

    cost, batches = histogram_greedy_batching(
        L, Bmax, init_bins=5, binning="quantile", assume_sorted=True
    )
    print("Approx total cost:", cost)
    print("Batches (count,sum,max,[left_idx,right_idx]):")
    for b in batches:
        print(b["count"], b["sum"], b["max"], [b["left_idx"], b["right_idx"]])
```
## Comparis  Test:
```python

import random
import time
from itertools import accumulate
from math import inf
from typing import List, Tuple, Optional, Literal
import numpy as np
import heapq


# ---------------------------
# Optimal DP (Problem A)
# ---------------------------
def minimize_padding_with_bmax_batches(
    lengths: List[int], Bmax: int, assume_sorted: bool = True
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Exact DP for Problem A. Returns (min_cost, partitions) where partitions are 1-based (start, end).
    Complexity: O(Bmax * N^2).
    """
    if not lengths:
        return 0, []
    L = lengths if assume_sorted else sorted(lengths)
    n = len(L)

    # Prefix sums
    S = [0] + list(accumulate(L))

    # dp[b][i] = min cost to cover first i items with exactly b batches
    dp = [[inf] * (n + 1) for _ in range(Bmax + 1)]
    prev = [[-1] * (n + 1) for _ in range(Bmax + 1)]
    dp[0][0] = 0

    for b in range(1, Bmax + 1):
        for i in range(1, n + 1):
            max_len = L[i - 1]  # nondecreasing; max in j..i is L[i-1]
            best_val = inf
            best_j = -1
            # enumerate start j
            for j in range(1, i + 1):
                batch_size = i - j + 1
                batch_sum = S[i] - S[j - 1]
                cost = batch_size * max_len - batch_sum
                cand = dp[b - 1][j - 1] + cost
                if cand < best_val:
                    best_val = cand
                    best_j = j
            dp[b][i] = best_val
            prev[b][i] = best_j

    # choose best b <= Bmax
    best_cost = inf
    best_b = -1
    for b in range(1, Bmax + 1):
        if dp[b][n] < best_cost:
            best_cost = dp[b][n]
            best_b = b

    # reconstruct
    partitions = []
    i = n
    b = best_b
    while b > 0 and i > 0:
        j = prev[b][i]
        partitions.append((j, i))
        i = j - 1
        b -= 1
    partitions.reverse()
    return int(best_cost), partitions


def cost_from_partitions_sorted(L: List[int], parts: List[Tuple[int, int]]) -> int:
    """Compute padding cost given sorted L and 1-based partitions."""
    cost = 0
    for (j, i) in parts:
        seg = L[j - 1 : i]
        m = seg[-1]
        c = len(seg)
        s = sum(seg)
        cost += c * m - s
    return int(cost)


# ---------------------------
# Histogram + Greedy merging (Sub-Optimal)
# ---------------------------
def histogram_greedy_batching(
    lengths: List[int],
    Bmax: int,
    init_bins: int = 100,
    binning: Literal["equalwidth", "quantile"] = "quantile",
    max_tokens: Optional[int] = None,
    assume_sorted: bool = True,
):
    """
    Sub-optimal: bucket-by-histogram, then greedily merge adjacent bins minimizing Δ cost until <= Bmax.
    Returns (approx_total_cost, batches), where batches are dicts with stats and [left_idx,right_idx] on sorted L.
    """
    if len(lengths) == 0:
        return 0, []
    L = np.asarray(lengths, dtype=np.int64)
    if not assume_sorted:
        L = np.sort(L)
    n = L.shape[0]
    Bmax = max(1, min(Bmax, n))

    # Build bin edges
    if binning == "equalwidth":
        lo, hi = int(L[0]), int(L[-1])
        if lo == hi:
            edges = np.array([lo, hi + 1.0], dtype=float)
        else:
            edges = np.linspace(lo, hi, init_bins + 1, dtype=float)
    elif binning == "quantile":
        qs = np.linspace(0, 1, init_bins + 1)
        edges = np.quantile(L, qs)
        edges = np.unique(edges)
        if edges.size == 1:
            edges = np.array([edges[0], edges[0] + 1.0], dtype=float)
    else:
        raise ValueError("binning must be 'equalwidth' or 'quantile'")

    # Assign items to bins (edges are ascending; right-exclusive bins)
    bin_idx = np.clip(np.digitize(L, edges[1:-1], right=False), 0, edges.size - 2)

    # Build non-empty bin segments
    bins = []
    start = 0
    for b in range(edges.size - 1):
        while start < n and bin_idx[start] < b:
            start += 1
        end = start
        while end < n and bin_idx[end] == b:
            end += 1
        if start == end:
            continue
        seg = L[start:end]
        c = int(seg.size)
        s = int(seg.sum())
        m = int(seg[-1])
        bins.append(
            {
                "count": c,
                "sum": s,
                "max": m,
                "left_idx": start,
                "right_idx": end - 1,
                "left_edge": float(edges[b]),
                "right_edge": float(edges[b + 1]),
                "alive": True,
                "left": None,
                "right": None,
                "id": b,
                "rev": 0,
            }
        )
        start = end

    if len(bins) == 0:
        return 0, []
    if len(bins) <= Bmax:
        total_cost = sum(b["count"] * b["max"] - b["sum"] for b in bins)
        return int(total_cost), bins

    # neighbor linkage
    for i in range(len(bins)):
        bins[i]["left"] = i - 1 if i > 0 else None
        bins[i]["right"] = i + 1 if i < len(bins) - 1 else None

    def bin_cost(b):
        return b["count"] * b["max"] - b["sum"]

    def merge_delta(i_left, i_right):
        if i_left is None or i_right is None:
            return None
        A = bins[i_left]
        B = bins[i_right]
        if (not A["alive"]) or (not B["alive"]):
            return None
        merged_count = A["count"] + B["count"]
        merged_max = max(A["max"], B["max"])
        if (max_tokens is not None) and (merged_count * merged_max > max_tokens):
            return None
        merged_sum = A["sum"] + B["sum"]
        cost_merge = merged_count * merged_max - merged_sum
        delta = cost_merge - (bin_cost(A) + bin_cost(B))
        return delta

    heap = []

    def push_pair(i_left, i_right):
        if i_left is None or i_right is None:
            return
        d = merge_delta(i_left, i_right)
        if d is None:
            return
        entry = (d, i_left, i_right, bins[i_left]["rev"], bins[i_right]["rev"])
        heapq.heappush(heap, entry)

    for i in range(len(bins) - 1):
        push_pair(i, i + 1)

    alive_count = len(bins)
    while alive_count > Bmax and heap:
        d, iL, iR, revL, revR = heapq.heappop(heap)
        if not (0 <= iL < len(bins) and 0 <= iR < len(bins)):
            continue
        A = bins[iL]
        B = bins[iR]
        if (not A["alive"]) or (not B["alive"]):
            continue
        if A["rev"] != revL or B["rev"] != revR:
            continue
        # re-check delta
        curr_d = merge_delta(iL, iR)
        if curr_d is None:
            continue

        # merge B into A
        A["count"] += B["count"]
        A["sum"] += B["sum"]
        A["max"] = max(A["max"], B["max"])
        A["right_idx"] = max(A["right_idx"], B["right_idx"])
        A["right_edge"] = max(A["right_edge"], B["right_edge"])
        A["rev"] += 1

        right_of_B = B["right"]
        A["right"] = right_of_B
        if right_of_B is not None:
            bins[right_of_B]["left"] = iL

        B["alive"] = False
        alive_count -= 1

        # update neighbors
        push_pair(A["left"], iL)
        push_pair(iL, A["right"])

    batches = [b for b in bins if b["alive"]]
    total_cost = sum(bin_cost(b) for b in batches)
    return int(total_cost), batches


def greedy_batches_to_partitions(batches, n):
    """
    Convert alive greedy batches with [left_idx,right_idx] (0-based) into partitions (1-based).
    Ensures they cover [1..n] in order (assuming bins were built from sorted L).
    """
    parts = []
    for b in batches:
        parts.append((b["left_idx"] + 1, b["right_idx"] + 1))
    parts.sort()
    # Optional: coalesce adjacent if any gap (shouldn't happen if built correctly)
    return parts


# ---------------------------
# Test Case Generators
# ---------------------------
def gen_uniform(n: int, lo: int, hi: int) -> List[int]:
    return sorted(random.randint(lo, hi) for _ in range(n))

def gen_heavy_tail(n: int, base: int = 3, maxpow: int = 12) -> List[int]:
    # Geometric-like / power-law-ish by exponentiating a uniform integer
    arr = [random.randint(1, base ** random.randint(0, maxpow)) for _ in range(n)]
    return sorted(arr)

def gen_mixture(n: int) -> List[int]:
    # Mix short and a few very long sequences
    arr = []
    for _ in range(n):
        if random.random() < 0.85:
            arr.append(random.randint(1, 64))
        elif random.random() < 0.95:
            arr.append(random.randint(65, 512))
        else:
            arr.append(random.randint(513, 4096))
    return sorted(arr)


# ---------------------------
# Benchmark Harness
# ---------------------------
def benchmark_case(L: List[int], Bmax: int, init_bins: int = 100, binning: str = "quantile"):
    # Optimal
    t0 = time.perf_counter()
    opt_cost, opt_parts = minimize_padding_with_bmax_batches(L, Bmax, assume_sorted=True)
    t1 = time.perf_counter()

    # Greedy
    t2 = time.perf_counter()
    approx_cost, greedy_batches = histogram_greedy_batching(
        L, Bmax, init_bins=init_bins, binning=binning, assume_sorted=True
    )
    # If you want to verify coverage, convert to 1-based partitions and recompute cost:
    greedy_parts = greedy_batches_to_partitions(greedy_batches, len(L))
    approx_cost_check = cost_from_partitions_sorted(L, greedy_parts)
    assert approx_cost == approx_cost_check, "Greedy cost mismatch vs recompute."
    t3 = time.perf_counter()

    # Gap
    gap_abs = approx_cost - opt_cost
    gap_pct = (gap_abs / opt_cost * 100.0) if opt_cost > 0 else 0.0

    return {
        "N": len(L),
        "Bmax": Bmax,
        "opt_cost": opt_cost,
        "opt_time_ms": (t1 - t0) * 1000,
        "approx_cost": approx_cost,
        "approx_time_ms": (t3 - t2) * 1000,
        "gap_abs": gap_abs,
        "gap_pct": gap_pct,
        "init_bins": init_bins,
        "binning": binning,
    }


def pretty_print_result(title: str, res: dict):
    print(f"\n=== {title} ===")
    print(f"N={res['N']}, Bmax={res['Bmax']}, bins={res['init_bins']} ({res['binning']})")
    print(f"Optimal   : cost={res['opt_cost']:<10d} time={res['opt_time_ms']:.2f} ms")
    print(f"Sub-opt   : cost={res['approx_cost']:<10d} time={res['approx_time_ms']:.2f} ms")
    print(f"Gap       : +{res['gap_abs']}  ({res['gap_pct']:.2f} %)")
    if res['opt_time_ms'] < res['approx_time_ms']:
        print(f"Time increase       : +{res['approx_time_ms'] / res['opt_time_ms']:.2f}x")
    else:
        print(f"Time decrease       : -{res['opt_time_ms'] / res['approx_time_ms']:.2f}x")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Define a suite of cases with increasing sizes.
    # Keep sizes moderate so the O(B*N^2) DP can finish quickly.
    cases = [
        ("Uniform small",   lambda: gen_uniform(80, 1, 128),  Bmax := 4),
        ("Uniform medium",  lambda: gen_uniform(160, 1, 256), Bmax := 6),
        ("Uniform High",  lambda: gen_uniform(1000, 1, 256), Bmax := 6),
        ("HeavyTail small", lambda: gen_heavy_tail(100, 3, 10), Bmax := 4),
        ("HeavyTail med",   lambda: gen_heavy_tail(200, 3, 12), Bmax := 6),
        ("HeavyTail High",  lambda: gen_heavy_tail(1000, 1, 256), Bmax := 6)
    ]

    init_bins = 60
    binning = "quantile"  # or "equalwidth"

    for name, gen_fn, bmax in cases:
        L = gen_fn()
        res = benchmark_case(L, bmax, init_bins=init_bins, binning=binning)
        pretty_print_result(name, res)


```

Results

```
=== Uniform small ===
N=80, Bmax=4, bins=60 (quantile)
Optimal   : cost=1029       time=1.61 ms
Sub-opt   : cost=1036       time=8.70 ms
Gap       : +7  (0.68 %)
Time increase       : +5.41x

=== Uniform medium ===
N=160, Bmax=6, bins=60 (quantile)
Optimal   : cost=2733       time=9.48 ms
Sub-opt   : cost=2763       time=0.62 ms
Gap       : +30  (1.10 %)
Time decrease       : -15.27x

=== Uniform High ===
N=1000, Bmax=6, bins=60 (quantile)
Optimal   : cost=19882      time=354.22 ms
Sub-opt   : cost=21943      time=0.59 ms
Gap       : +2061  (10.37 %)
Time decrease       : -595.73x

=== HeavyTail small ===
N=100, Bmax=4, bins=60 (quantile)
Optimal   : cost=298343     time=1.97 ms
Sub-opt   : cost=356806     time=0.38 ms
Gap       : +58463  (19.60 %)
Time decrease       : -5.22x

=== HeavyTail med ===
N=200, Bmax=6, bins=60 (quantile)
Optimal   : cost=2237145    time=12.48 ms
Sub-opt   : cost=2807041    time=0.40 ms
Gap       : +569896  (25.47 %)
Time decrease       : -31.06x

=== HeavyTail High ===
N=1000, Bmax=6, bins=60 (quantile)
Optimal   : cost=0          time=291.42 ms
Sub-opt   : cost=0          time=0.29 ms
Gap       : +0  (0.00 %)
Time decrease       : -1019.75x

```
