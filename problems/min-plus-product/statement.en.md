---
title: One Change
---

A bus runs from each of the `N` villages of the valley to each of the `N` railway stations, and a train runs from
each station to each of the `N` towns on the coast. The travel agency Pereval is compiling a guide: the price of the
cheapest way from every village to every town — by bus to some station and on by train.

The prices are written in two matrices of `N` rows and `N` columns, each stored row by row: the bus ticket from
village `i` to station `k` costs `A[i * N + k]`, the train ticket from station `k` to town `j` costs `B[k * N + j]`.
Find the matrix `C` of the same size:

```
C[i * N + j] = the minimum of A[i * N + k] + B[k * N + j] over all 0 ≤ k < N
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const int* A, const int* B, int* C, int N);
```

`A`, `B` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. This is a matrix product with the multiplication replaced by an addition and the addition
by a minimum: N³ sums for N² answers. There is no ready-made library for such a product, and the order in which `i`,
`j` and `k` are walked decides how memory is read.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.
A matrix of the largest test takes 64 KiB and does not fit the L1 cache (32 KiB): a walk down a column of `B` is
a miss at every step.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.
The three matrices of the largest test take 27 648 bytes of the 32 KB: there is no room for a copy of a matrix.

## Example

| N | A | B | C |
| --- | --- | --- | --- |
| 3 | 4 9 2 7 1 8 5 5 5 | 3 6 1 2 2 9 8 1 4 | 7 3 5 3 3 8 7 6 6 |

The same matrices row by row:

```
A:  4 9 2      B:  3 6 1      C:  7 3 5
    7 1 8          2 2 9          3 3 8
    5 5 5          8 1 4          7 6 6
```

From village 0 to town 1 the cheapest way is through station 2: the bus for 2 and the train for 1, 3 in total.
Station 0 would make it 4 + 6 = 10, station 1 would make it 9 + 2 = 11. Station 1 is the closest one for village 1,
the bus there costs 1, and the trips to towns 0 and 1 go through it for 3. But the train from it to town 2 costs 9,
so that trip is cheaper through station 0: 7 + 1 = 8.

## Constraints

- 1 ≤ N ≤ 1024 on the GPU, 1 ≤ N ≤ 128 on `riscv-rvv` and `cuda-sim`, 1 ≤ N ≤ 48 on `coral-npu`;
- 0 ≤ A[i * N + k], B[k * N + j] ≤ 1 000 000;
- the answer is compared exactly.
