---
title: Sneezewort
---

A sneezewort seed was planted in the botanical garden. The plant grows by a strict rule: every shoot spends the first
week of its life gaining strength and then puts out one new shoot every week. A week after planting the first shoot
came up, after two weeks it was still alone, after three there were two shoots, after four — three, after five —
five. The gardener keeps count in a 32-bit counter: once the number of shoots no longer fits, the upper bits are
simply lost.

The number of shoots after `n` weeks is the Fibonacci number `F(n)`:

```
F(0) = 0, F(1) = 1, F(n) = F(n - 1) + F(n - 2)
```

You are given an array `A` of `N` integers. For each of them find what the counter shows `A[i]` weeks after
planting:

```
C[i] = F(A[i]) mod 2^32
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint32_t* A, uint32_t* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. Adding week after week takes too long on any processor: `A[i]` goes up to a billion. But a pair of
neighbours `F(k)` and `F(k + 1)` gives the pair `F(2k)` and `F(2k + 1)` at once, and then the billionth week is
about thirty steps away.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function. Unsigned 32-bit addition and multiplication drop the upper bits by themselves, so
no 64-bit numbers are needed here.

## Example

| A | C |
| --- | --- |
| 0 1 2 3 10 47 48 1000000000 | 0 1 1 2 55 2971215073 512559680 1532868155 |

After 10 weeks there are 55 shoots in the bed. `F(47)` = 2 971 215 073 is the last Fibonacci number that fits
32 bits. Already `F(48)` = 4 807 526 976, and the counter shows 4 807 526 976 − 4 294 967 296 = 512 559 680.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`, 1 ≤ N ≤ 3000 on `coral-npu`;
- 0 ≤ A[i] ≤ 1 000 000 000;
- the answer is compared exactly.
