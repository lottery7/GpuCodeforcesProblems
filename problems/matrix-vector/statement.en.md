---
title: Matrix Times Vector
---

Multiply a matrix by a vector: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given a matrix `A` of `N` rows and `M` columns stored row by row and a vector `X` of `M` integers, calculate

```
Y[i] = A[i][0] * X[0] + A[i][1] * X[1] + ... + A[i][M-1] * X[M-1]
```

The element `A[i][j]` is `A[i * M + j]`. The products and the sums are 64-bit: a million times a million does not fit
an `int`.

Implement the function

```cpp
void solve(const int* A, const int* X, int64_t* Y, int N, int M);
```

`A`, `X` and `Y` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. This is one layer of a neural network without the activation: `N` reductions that share the
vector `X`.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.
A product of two 32-bit numbers widens to 64 bits: half as many lanes per register.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.
A 64-bit sum takes two registers of this 32-bit core, and its vectors have no 64-bit lanes.

## Example

| N | M | A | X | Y |
| --- | --- | --- | --- | --- |
| 2 | 3 | 1 2 3 4 5 6 | 1 0 -1 | -2 -2 |

## Constraints

- 1 ≤ N, M ≤ 4096 on the GPU, N, M ≤ 512 on `riscv-rvv`, N * M ≤ 51 200 on `cuda-sim`, N * M ≤ 6000 on `coral-npu`;
- -1 000 000 ≤ A[i][j], X[j] ≤ 1 000 000;
- the answer is compared exactly.
