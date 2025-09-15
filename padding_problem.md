# Problem — Minimize Padding Cost with at most `Bmax` Batches

## Problem Statement
You are given an array `L[1..N]` of **nondecreasing** positive integers (token lengths). You must partition the array into at most `Bmax` **contiguous** groups (keeping order).  
For any group `L[j..i]`, define its padding cost as:

**cost(j,i) = (i-j+1) × max(L[j..i]) - Σ(L[k] for k=j to i)**

Because `L` is nondecreasing, max(L[j..i]) = L[i].

**Goal:** minimize the total padding cost across all groups, subject to using at most `Bmax` groups.

**Input:** `N`, `Bmax`, and a nondecreasing array `L[1..N]`  
**Output:** the minimal total cost; optionally, one optimal partition.

---

## My Approach: Two Different Strategies

### 1) Dynamic Programming (Exact Optimal)
Let `S[i] = L[1]+...+L[i]` (prefix sums, with `S[0]=0`).

Define `dp[b][i]` = minimal total cost to cover the **first `i` items** using **exactly `b` groups**.

Here's how I think about it:
- `dp[b][i]` = "What's the cheapest way to batch the first `i` sequences using exactly `b` batches?"
- To fill `dp[b][i]`, I try every possible starting point `j` for the last batch

Let me set up some notation first. I'll use prefix sums `S[i]` to quickly calculate the sum of any range:
- `S[i] = L[1] + L[2] + ... + L[i]`
- `S[0] = 0`

Now the recurrence becomes cleaner:
\[
dp[b][i] = \min_{1\le j\le i}\Big( dp[b-1][j-1] + (i-j+1)\cdot L[i] - (S[i]-S[j-1]) \Big)
\]

The intuition: I'm considering making the last batch span from position `j` to `i`. The cost of this batch is `(number of items) × (max length) - (sum of lengths)`. Since everything's sorted, `L[i]` is the max.

**Base case:** `dp[0][0] = 0` (zero cost for zero items in zero batches)  
**Answer:** `min(dp[1][N], dp[2][N], ..., dp[Bmax][N])` 

The time complexity is `O(Bmax × N²)` - not terrible for moderate sizes, but it can get expensive when `N` gets large.

**Reconstruction trick:** I keep track of which `j` gave me the minimum for each `dp[b][i]`. Then I can backtrack to recover the actual partition.

**Why this works:** The key insight is that since our lengths are sorted, I never need to worry about non-contiguous partitions. The optimal solution will always use contiguous chunks, and DP naturally finds the best cut points.

---

## Python Solution (DP with Reconstruction)

```python
from math import inf
from itertools import accumulate
from typing import List, Tuple

def minimize_padding_with_bmax_batches(lengths: List[int], Bmax: int, assume_sorted: bool = True
                                      ) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Find the optimal way to batch sequences to minimize padding waste.
    
    This is the "proper" DP solution - guaranteed optimal but can be slow for large inputs.
    
    Args:
        lengths: List of sequence lengths (should be sorted ascending)
        Bmax: Maximum number of batches allowed
        assume_sorted: Set False if you need me to sort the input first
        
    Returns:
        (total_padding_cost, partitions) where partitions are 1-indexed ranges
    """
    if not lengths:
        return 0, []

    # Make sure we have sorted input - this is crucial for the algorithm
    L = lengths if assume_sorted else sorted(lengths)
    n = len(L)

    # Build prefix sums for fast range sum queries
    # S[i] = sum of L[0] through L[i-1], so S[0] = 0
    S = [0] + list(accumulate(L))

    # The heart of the DP:
    # dp[b][i] = minimum cost to batch first i sequences using exactly b batches
    dp = [[inf] * (n + 1) for _ in range(Bmax + 1)]
    # Keep track of decisions for reconstruction
    prev = [[-1] * (n + 1) for _ in range(Bmax + 1)]  
    
    # Base case: zero sequences, zero batches, zero cost
    dp[0][0] = 0

    # Fill the DP table
    for b in range(1, Bmax + 1):  # for each possible number of batches
        for i in range(1, n + 1):  # for each ending position
            # Try every possible starting position j for the last batch
            max_len = L[i - 1]  # max length in any batch ending at i (since sorted)
            best_cost = inf
            best_start = -1
            
            for j in range(1, i + 1):  # j is 1-indexed start of last batch
                # Cost of batch from j to i
                batch_size = i - j + 1
                batch_sum = S[i] - S[j - 1]  # sum of L[j-1] to L[i-1]
                batch_cost = batch_size * max_len - batch_sum
                
                # Total cost = cost of first j-1 items + this batch
                total_cost = dp[b - 1][j - 1] + batch_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_start = j
                    
            dp[b][i] = best_cost
            prev[b][i] = best_start

    # Find the best solution using at most Bmax batches
    best_cost = inf
    best_num_batches = -1
    for b in range(1, Bmax + 1):
        if dp[b][n] < best_cost:
            best_cost = dp[b][n]
            best_num_batches = b

    # Backtrack to reconstruct the actual partition
    partitions = []
    i = n
    b = best_num_batches
    while b > 0 and i > 0:
        j = prev[b][i]
        partitions.append((j, i))  # 1-based range [j, i]
        i = j - 1
        b -= 1
    partitions.reverse()  # we built it backwards

    return int(best_cost), partitions

# Let's test it on our example
if __name__ == "__main__":
    # My favorite test case
    L = [1, 2, 3, 3, 3, 6, 10, 19]  
    Bmax = 2
    
    cost, parts = minimize_padding_with_bmax_batches(L, Bmax)
    print(f"Minimum padding cost: {cost}")
    print(f"Optimal batches: {parts}")
    
    # Let's verify this makes sense:
    # Batch 1: positions 1-5 = [1,2,3,3,3], max=3, cost = 5*3 - 12 = 3
    # Batch 2: positions 6-8 = [6,10,19], max=19, cost = 3*19 - 35 = 22  
    # Total: 3 + 22 = 25 ✓
```

# Strategy 2: When You Need Speed Over Perfection

## The Reality Check

Okay, so the DP solution is mathematically beautiful and gives you the optimal answer. But here's the thing - when you're dealing with millions of sequences, that O(Bmax × N²) starts to hurt. 
So I came up with a faster approximation that's "good enough" for most practical cases. The idea is pretty intuitive:

1. **Group similar lengths together** - Use histograms to bucket sequences by length
2. **Merge smartly** - Greedily combine adjacent buckets, always picking the merge that adds the least padding waste

Think of it like organizing your bookshelf. You don't need to consider every possible arrangement - just group books of similar height together, then merge shelves when you run out of space.

## How The Histogram Approach Works

### The Basic Insight

Instead of considering all N² possible ways to split the sequences, I first create "natural" groups using histograms. Sequences with similar lengths end up in the same bucket, which already minimizes padding within each bucket.

For any batch containing:
- `c` sequences
- total length sum `s`  
- maximum length `m`

The padding cost is still: `cost = c × m - s`

### The Greedy Merging Strategy

Once I have my initial buckets, I need to reduce them down to at most `Bmax` batches. Here's where the greedy magic happens:

For any two adjacent buckets A and B, if I merge them:
- New count: `c = cA + cB`
- New sum: `s = sA + sB`
- New max: `m = max(mA, mB)`

The extra padding I introduce is:

**Δ = cost_after_merge - (cost_A + cost_B)**

I always merge the pair with the smallest Δ - the one that hurts the least.

**Why only adjacent buckets?** Since I sorted everything by length, adjacent buckets have the most similar lengths. Merging distant buckets would create huge padding gaps.

---

## The Step-by-Step Process

Here's exactly what I do:

1. **Sort the input** (if it's not already sorted)
2. **Create initial buckets** using histograms:
   - **Equal-width bins**: Divide the range [min_length, max_length] into K equal parts
   - **Quantile bins** (my preference): Place bin boundaries at quantiles so each bin has roughly the same number of items
3. **Calculate stats** for each non-empty bucket: count, sum, and max length
4. **Build a priority queue** of all possible adjacent merges, sorted by their Δ cost
5. **Greedily merge** until I have ≤ Bmax buckets:
   - Pop the cheapest merge from the heap
   - Combine those two buckets
   - Update the heap with new merge costs involving the combined bucket
6. **Done!** The remaining buckets are my final batches

### Complexity Analysis (Why This Is Fast)

- **Bucketing**: O(N) - NumPy's quantile computation is nearly linear in practice
- **Stats computation**: O(N) - one pass through the data
- **Merging**: O(K log K) where K is initial bins - much smaller than N usually
- **Overall**: Way better than O(Bmax × N²) for large N

### When This Works Well vs. When It Struggles

**Works great when:**
- Your length distribution is reasonably smooth (no crazy outliers)
- You have enough initial bins to capture the natural groupings
- You're okay with ~5-15% suboptimality for massive speed gains

**Struggles when:**
- You have weird bimodal distributions with huge gaps
- Your data has lots of identical lengths (creates degenerate bins)
- You really need that last 1% of optimality

**Pro tip:** Start with 50-200 initial bins. More bins = better approximation but slower merging.

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
    The "fast and good enough" solution for large datasets.
    
    Strategy: Create lots of small buckets first, then greedily merge adjacent ones
    until we hit our batch limit. Much faster than DP but slightly suboptimal.
    
    Args:
        lengths: Your sequence lengths (should be sorted, or set assume_sorted=False)
        Bmax: Maximum batches you want in the end
        init_bins: How many buckets to start with (more = better quality, slower merging)
        binning: "quantile" puts equal counts in each bin, "equalwidth" uses equal ranges
        max_tokens: Optional limit on batch_size * max_length per batch
        assume_sorted: Set False if your input isn't sorted yet
        
    Returns:
        (total_padding_cost, batch_info_list)
        Each batch has stats like count, sum, max, and which indices it covers
    """
    if len(lengths) == 0:
        return 0, []

    # Step 1: Make sure we have sorted data
    L = np.asarray(lengths, dtype=np.int64)
    if not assume_sorted:
        L = np.sort(L)
    n = L.shape[0]

    # Can't have more batches than sequences!
    Bmax = max(1, min(Bmax, n))

    # Step 2: Create the initial histogram bins
    if binning == "equalwidth":
        # Divide the range [min, max] into equal-width intervals
        lo, hi = int(L[0]), int(L[-1])
        if lo == hi:
            # Edge case: all lengths are identical
            edges = np.array([lo, hi + 1], dtype=float)
        else:
            edges = np.linspace(lo, hi, init_bins + 1, dtype=float)
            
    elif binning == "quantile":
        # My preferred approach: put bin edges at quantiles
        # This ensures each bin gets roughly the same number of items
        qs = np.linspace(0, 1, init_bins + 1)
        edges = np.quantile(L, qs)
        
        # Handle ties by removing duplicate edges
        edges = np.unique(edges)
        if edges.size == 1:
            # Another edge case: everything has the same length
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
# Battle of the Algorithms: DP vs Histogram Greedy

Let me put these two approaches head-to-head and see how they actually perform in practice. I've set up some test cases that should reveal the trade-offs between optimal quality and speed.
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
# Test Data Generators
# ---------------------------

def gen_uniform(n: int, lo: int, hi: int) -> List[int]:
    """Nice, well-behaved uniform distribution - should be easy for both algorithms"""
    return sorted(random.randint(lo, hi) for _ in range(n))

def gen_heavy_tail(n: int, base: int = 3, maxpow: int = 12) -> List[int]:
    """The nasty case - lots of short sequences + a few monsters
    This is where histogram bucketing might struggle"""
    arr = [random.randint(1, base ** random.randint(0, maxpow)) for _ in range(n)]
    return sorted(arr)

def gen_mixture(n: int) -> List[int]:
    """Real-world-ish: mostly short (85%), some medium (10%), few long (5%)
    Kind of like what you'd see in NLP datasets"""
    arr = []
    for _ in range(n):
        if random.random() < 0.85:
            arr.append(random.randint(1, 64))        # Short sequences
        elif random.random() < 0.95:
            arr.append(random.randint(65, 512))      # Medium sequences  
        else:
            arr.append(random.randint(513, 4096))    # Long sequences
    return sorted(arr)


# ---------------------------
# Benchmark Harness
# ---------------------------
def benchmark_case(L: List[int], Bmax: int, init_bins: int = 100, binning: str = "quantile"):
    """Run both algorithms on the same data and compare results"""
    
    # Time the optimal DP solution
    print(f"  Running DP (optimal)...", end=" ", flush=True)
    t0 = time.perf_counter()
    opt_cost, opt_parts = minimize_padding_with_bmax_batches(L, Bmax, assume_sorted=True)
    t1 = time.perf_counter()
    print(f"done in {(t1-t0)*1000:.1f}ms")

    # Time the histogram greedy approximation
    print(f"  Running histogram greedy...", end=" ", flush=True)
    t2 = time.perf_counter()
    approx_cost, greedy_batches = histogram_greedy_batching(
        L, Bmax, init_bins=init_bins, binning=binning, assume_sorted=True
    )
    t3 = time.perf_counter()
    print(f"done in {(t3-t2)*1000:.1f}ms")
    
    # Double-check our greedy cost calculation (paranoid but good practice)
    greedy_parts = greedy_batches_to_partitions(greedy_batches, len(L))
    approx_cost_check = cost_from_partitions_sorted(L, greedy_parts)
    assert approx_cost == approx_cost_check, "Greedy cost calculation bug!"

    # Calculate the quality gap
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

# What I Learned From The Benchmarks

Here's what happened when I ran both algorithms on different types of data:

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

## My Takeaways

**The crossover point is around N=100-200:** For small datasets, DP is actually faster AND optimal. But once you hit a few hundred sequences, the histogram approach starts crushing it on speed.

**Uniform data is forgiving:** Even at 10% suboptimality, the histogram method gives you 500x speedup. That's usually worth it unless you're really optimizing for the last penny.

**Heavy-tailed data is where it gets interesting:** The gap can grow to 25%+ because the histogram struggles with outliers. Those few monster sequences mess up the bucketing strategy.

**That weird "cost=0" case:** This happens when all sequences are identical - both algorithms recognize that one batch is optimal, so there's no padding cost at all.

**For production systems:** I'd probably use DP for N < 500 and histogram for larger datasets, maybe with some hybrid approach where I run DP on a sample first to estimate the quality gap.

The bottom line? If you're processing millions of sequences in real-time, take the 15% hit and get your results 1000x faster. If you're doing offline optimization where every token counts, stick with DP.
