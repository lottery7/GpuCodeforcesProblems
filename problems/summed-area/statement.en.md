---
title: Census in Gridville
---

The blocks of the city of Gridville form a table of `N` rows and `M` columns: the rows run from north to south, the
columns from west to east. After the census, the change of the population over the year is known for every block:
the number is positive if the block gained residents and negative if it lost them. The city hall needs the same
report by district: for every block, the total change over all blocks that lie no farther south and no farther east.

You are given a table `A` of `N` rows and `M` columns stored row by row: the number in row `r` and column `c` is
`A[r * M + c]`. Fill the table `S` of the same size:

```
S[i * M + j] = the sum of A[r * M + c] over all 0 ≤ r ≤ i and 0 ≤ c ≤ j
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const int* A, int* S, int N, int M);
```

`A` and `S` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. This is a prefix sum in two directions at once. Along a row every sum depends on the one before it, but
the rows do not depend on each other, and the same holds for the columns: what is left to decide is which part to
compute in turn and which at the same time.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.
Adding the finished row above to a row is one vector operation; a sum along the row does not fit a vector that
easily.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` and `M` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve`
taking `int` would be another function.
The two tables of the largest test take 26 880 bytes of the 32 KB: there is no room for a third one.

## Example

| N | M | A | S |
| --- | --- | --- | --- |
| 3 | 4 | 3 -1 4 0 -2 5 -6 2 1 0 7 -3 | 3 2 6 6 1 5 3 5 2 6 11 10 |

The same tables row by row:

```
A:  3 -1  4  0      S:  3  2  6  6
   -2  5 -6  2          1  5  3  5
    1  0  7 -3          2  6 11 10
```

The top row of `S` is the plain prefix sums of the top row of `A`. The number `S[1 * 4 + 2]` = 3 is made of the six
blocks in rows 0–1 and columns 0–2: 3 − 1 + 4 − 2 + 5 − 6. The last number, 10, is the sum of the whole table.

## Constraints

- N, M ≥ 1; N × M ≤ 16 777 216 on the GPU, N × M ≤ 100 000 on `riscv-rvv` and `cuda-sim`, N × M ≤ 3360 on
  `coral-npu`;
- −100 ≤ A[r * M + c] ≤ 100, so the absolute value of any sum is at most 1 677 721 600 and fits an `int`;
- the answer is compared exactly.
