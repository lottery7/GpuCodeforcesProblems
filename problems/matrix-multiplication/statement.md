---
code: "2"
title: Matrix Multiplication
difficulty: medium
tags: [matrix, arithmetic]
authors: []
---

Implement matrix multiplication on the GPU.

Given two dense matrices `A` (M×K) and `B` (K×N), stored row by row, compute the product
`C = A × B` (M×N, row by row).

```cpp
void solve(const float* A, const float* B, float* C, int M, int K, int N);
```

`A`, `B` and `C` point to GPU memory. `solve` runs on the CPU and launches kernels.
Your implementation must handle the tiling and memory access patterns efficiently.

## Example

`M = 2, K = 3, N = 2`, `A = [[1, 2, 3], [4, 5, 6]]`, `B = [[1, 0], [0, 1], [1, 1]]`,
answer `C = [[4, 5], [10, 11]]`.

## Constraints

- 1 ≤ M, K, N ≤ 1000;
- the answer is compared with absolute and relative tolerance `1e-3`.
