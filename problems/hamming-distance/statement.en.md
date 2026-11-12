---
title: Hamming Distance
---

Count the bits in which two bit strings differ: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a
circuit (`rtl`), on a vector RISC-V CPU (`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

The strings are given as two arrays `A` and `B` of `N` 32-bit words. Calculate

```
R[0] = popcount(A[0] ^ B[0]) + popcount(A[1] ^ B[1]) + ... + popcount(A[N-1] ^ B[N-1])
```

where `popcount(x)` is the number of one bits of `x`.

Implement the function

```cpp
void solve(const uint32_t* A, const uint32_t* B, uint32_t* R, int N);
```

`A`, `B` and `R` point to GPU memory; `R` holds one element. `solve` runs on the CPU: it launches the kernels and
returns, the judge synchronizes afterwards.

Counting bits is one instruction on some processors and a small program on others: look at what yours has.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. The judge raises `start` for the first cycle and from the next
cycle on feeds one pair of words per cycle: `A_data` and `B_data` hold `A[i]` and `B[i]` while `A_valid` and
`B_valid` are high. Hand the answer back by raising `R_valid` for one cycle with the count on `R_data`, then raise
`done`. Registers are zero before `start`. The score is the number of cycles until `done` times the logic depth (the
most gates on a path between registers) times the transistors of the synthesized circuit: smaller is better. The
circuit must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
This processor has no bit-counting instruction, neither scalar nor vector. The score is the instructions executed
plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. Zbb counts the bits of a register in one instruction
(`cpop`); the vectors have no such instruction. The core has 8 KB of memory for code and 32 KB for data; the arrays
lie in that data memory. The score is the instructions the core executed: smaller is better. In C++ declare `N` as
`int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking `int` would be
another function.

## Example

| A | B | R |
| --- | --- | --- |
| 0 255 5 | 1 15 5 | 5 |

`0 ^ 1` has one bit set, `255 ^ 15 = 240` has four, `5 ^ 5` has none.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- the words take any 32-bit value;
- the answer is compared exactly.
