---
title: Heat Records
---

The weather station at Cape Winds has kept a log since the day it opened: every noon the air temperature goes into
it, in thousandths of a degree. The head of the station is fond of records and wants to know, for every day, the
highest temperature observed so far — from the first day up to and including that one.

You are given an array `A` of `N` integers: `A[i]` is the temperature on day `i`. For every day find

```
C[i] = max(A[0], A[1], ..., A[i])
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const int* A, int* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. The answer for day `i` depends on every day before it: on a GPU the loop "from the start to the end" has
to be cut into parts that can be computed at the same time.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the input `N` and two streams. The judge raises `start` for
the first cycle and from the next cycle on feeds one element per cycle: `A_data` holds `A[i]` while `A_valid` is
high. Hand `C[i]` back in the same order by raising `C_valid` with the element on `C_data`, then raise `done`. The
stream cannot be paused or rewound. Registers are zero before `start`. The score is the number of cycles until
`done` times the logic depth (the most gates on a path between registers) times the transistors of the synthesized
circuit: smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.

## Example

| A | C |
| --- | --- |
| -3500 1200 800 1200 4100 -200 | -3500 1200 1200 1200 4100 4100 |

On the first day the record is the only measurement, −3.5°. The second day is warmer, 1.2°, and that record stands
for three days until the fifth day brings 4.1°.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- −90 000 ≤ A[i] ≤ 60 000;
- the answer is compared exactly.
