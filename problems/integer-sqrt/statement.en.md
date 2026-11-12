---
title: Drone Square
---

At the festival of light the drones line up in the sky every evening as a solid square: `k` rows of `k` drones. The
director wants the square as large as possible; the drones that find no place in it stay on the ground. For each of
the `N` evenings it is known how many working drones the troupe has.

You are given an array `A` of `N` integers: `A[i]` is the number of drones on evening `i`. For every evening find
the side of the largest square, that is, the largest integer `k` with `k * k ≤ A[i]`:

```
C[i] = floor(sqrt(A[i]))
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint32_t* A, uint32_t* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. The answer must be exact, and the numbers use all 32 bits: the mantissa of a `float` has only 24, so next
to large perfect squares a single precision root is off by one. The tests do contain such numbers.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A` and `C` are memories of the judge: put an index on `A_addr`
and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`. The judge
raises `start` for the first cycle; raise `done` once every element of `C` is written. Registers are zero before
`start`. The score is the number of cycles until `done` times the logic depth (the most gates on a path between
registers) times the transistors of the synthesized circuit: smaller is better. The circuit must be synchronous: no
latches, no asynchronous resets. A root can be taken by the long-hand method, one binary digit per step: shifts,
a comparison and a subtraction are enough. How many such steps go into one cycle is up to you.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function. This core has single precision floating point only, and a 64-bit integer takes two
of its registers.

## Example

| A | C |
| --- | --- |
| 0 1 8 9 15 16 999999 1000000 4294836224 4294836225 4294967295 | 0 1 2 3 3 4 999 1000 65534 65535 65535 |

Eight drones make only a 2 × 2 square, and the ninth one allows 3 × 3. A million drones is exactly 1000 × 1000, but
let one of them fail to take off and the side is 999. The largest square of this problem is 65 535 × 65 535 =
4 294 836 225 drones: even 4 294 967 295 drones are not enough for a side of 65 536.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 2048 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- 0 ≤ A[i] ≤ 4 294 967 295;
- the answer is compared exactly.
