---
title: Dot Product
---

Compute the dot product of two vectors: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit
(`rtl`), on a vector RISC-V CPU (`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given two vectors `A` and `B` of `N` integers, calculate

```
R[0] = A[0] * B[0] + A[1] * B[1] + ... + A[N-1] * B[N-1]
```

Implement the function

```cpp
void solve(const int* A, const int* B, int64_t* R, int N);
```

`A`, `B` and `R` point to GPU memory; `R` holds one 64-bit element. `solve` runs on the CPU: it launches the kernels
and returns, the judge synchronizes afterwards. Unlike an element-wise operation, a sum has to bring all the
elements together: the puzzle is how.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. The judge raises `start` for the first cycle and from the next
cycle on feeds one pair per cycle: `A_data` and `B_data` hold `A[i]` and `B[i]` while `A_valid` and `B_valid` are
high. Hand the answer back by raising `R_valid` for one cycle with the sum on the 64 bits of `R_data`, then raise
`done`. Registers are zero before `start`. The score is the number of cycles until `done` times the logic depth (the
most gates on a path between registers) times the transistors of the synthesized circuit: smaller is better. A
multiplier is the most expensive part of this circuit; the constraints below say how wide it has to be. The circuit
must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. The core is
32-bit, so a 64-bit sum costs more here than a 32-bit one. In C++ declare `N` as `int32_t`, as the solution template
does: on this target `int32_t` is `long`, and a `solve` taking `int` would be another function.

## Example

| A | B | R |
| --- | --- | --- |
| 1 2 3 4 | 5 6 7 8 | 70 |

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 2000 on `coral-npu`;
- 0 ≤ A[i], B[i] ≤ 1000;
- the answer is compared exactly.
