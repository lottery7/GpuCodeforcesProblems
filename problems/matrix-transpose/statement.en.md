---
title: Matrix Transpose
---

Transpose a matrix: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU (`riscv-rvv`)
or on the Coral NPU core (`coral-npu`).

Given a matrix `A` of `N` rows and `M` columns, stored row by row, fill the matrix `B` of `M` rows and `N` columns,
stored row by row too:

```
B[j * N + i] = A[i * M + j],    0 ≤ i < N, 0 ≤ j < M
```

Implement the function

```cpp
void solve(const float* A, float* B, int N, int M);
```

`A` and `B` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns, the judge synchronizes
afterwards.

Nothing is computed here, the elements only move: the whole cost is in the order they are read and written.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` is not allowed in the solution, the data is already on the GPU.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the matrices are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better. The cache model
has a 32 KiB L1 and a 1 MiB L2 with 64-byte lines; the large matrices of the tests fit neither, so a walk that reads
rows and writes columns misses on almost every element.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
`A` and `B` lie in that data memory. There are no caches: the score is the instructions the core executed, smaller
is better, and a vector load or store may step through memory with any stride. In C++ declare `N` and `M` as
`int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking `int` would be
another function.

## Example

`N = 2`, `M = 3`:

| A | B |
| --- | --- |
| 1 2 3 4 5 6 | 1 4 2 5 3 6 |

## Constraints

- N × M ≤ 16 777 216 on the GPU, N × M ≤ 1 048 576 on `riscv-rvv`, N × M ≤ 51 200 on `cuda-sim`,
  N × M ≤ 3360 on `coral-npu`; N, M ≥ 1;
- the elements are in [0, 1);
- the answer is compared exactly.
