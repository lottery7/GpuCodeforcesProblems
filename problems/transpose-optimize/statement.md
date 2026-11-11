---
code: "9"
title: Speed Up the Transpose
difficulty: medium
tags: [matrix, cache, optimization]
authors: []
---

The editor holds a working transpose for a vector RISC-V CPU (`riscv-rvv`). It is correct and slow: make it cheaper.

`A` is a matrix of `N` rows and `M` columns stored row by row; `B` receives its transpose, `M` rows of `N` columns:

```
B[j * N + i] = A[i * M + j]
```

```cpp
void solve(const float* A, float* B, int32_t N, int32_t M);
```

The solution is C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors. The score is the instructions executed
plus 10 per L1 miss and 100 per L2 miss, and the page shows it next to the score of the original code: smaller is
better. The original loop executes few instructions per element - look at where the rest of its score comes from.
The cache model has a 32 KiB L1 and a 1 MiB L2 with 64-byte lines, and the measured matrices are larger than both.

## Example

| N | M | A | B |
| --- | --- | --- | --- |
| 2 | 3 | 1 2 3 4 5 6 | 1 4 2 5 3 6 |

## Constraints

- 1 ≤ N, M ≤ 1024;
- the answer is compared exactly.
