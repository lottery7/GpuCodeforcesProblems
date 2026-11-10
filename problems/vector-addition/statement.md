---
code: "1"
title: Vector Addition
difficulty: easy
tags: [vector, arithmetic]
authors: []
---

Implement vector addition on the GPU.

Given two input vectors `A` and `B` of `N` floats, calculate:

```
C[i] = A[i] + B[i]
```

Implement the function

```cpp
void solve(const float* A, const float* B, float* C, int N);
```

`A`, `B` and `C` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns,
the judge synchronizes afterwards. Your implementation must support all valid values of `N`.

## Example

| A | B | C |
| --- | --- | --- |
| 1 2 3 4 | 5 6 7 8 | 6 8 10 12 |

## Constraints

- 1 ≤ N ≤ 10 000 000;
- the answer is compared with absolute and relative tolerance `1e-6`.
