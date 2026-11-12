---
title: Double Exposure
---

In the photo lab two frames are printed onto one sheet through a mask: where the mask is black only the first frame
shows, where it is white only the second, and in the grey places the frames are mixed in the proportion the mask
gives. Vera the lab assistant wants to see the result on the screen before she spends any paper on it.

You are given three arrays of `N` bytes each: the frames `A` and `B` and the mask `T`. For every pixel find

```
C[i] = (A[i] * (255 - T[i]) + B[i] * T[i] + 127) / 255
```

The division is an integer one, the remainder is dropped; the term 127 turns it into rounding to the nearest
integer.

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint8_t* A, const uint8_t* B, const uint8_t* T, uint8_t* C, int N);
```

`A`, `B`, `T` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. The pixels do not depend on each other, so it all comes down to the price of one: the
intermediate values do not fit a byte, and the division is the most expensive operation of the formula, although
its divisor is always the same.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution. The
compiler folds a sum of two products of bytes into the `dp2a` instruction, which GPGPU-Sim does not have: a solution
with the formula written word for word gets `RUNTIME_ERROR` here. Rewrite it so that one multiplication is left.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the input `N` and four streams of bytes. The judge raises
`start` for the first cycle and from the next cycle on feeds one pixel per cycle: `A_data`, `B_data` and `T_data`
hold `A[i]`, `B[i]` and `T[i]` while `A_valid`, `B_valid` and `T_valid` are high. Hand `C[i]` back in the same order
by raising `C_valid` with the byte on `C_data`, then raise `done`. The stream cannot be paused or rewound. Registers
are zero before `start`. The score is the number of cycles until `done` times the logic depth (the most gates on
a path between registers) times the transistors of the synthesized circuit: smaller is better. The circuit must be
synchronous: no latches, no asynchronous resets. A multiplier and a divider are the most expensive things in such
a circuit: one multiplier is enough, and the divider can be left out altogether.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better. A product is
cramped in a vector of bytes: it has to be widened to 16 bits, and the answer narrowed back.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.

## Example

| A | B | T | C |
| --- | --- | --- | --- |
| 200 200 0 255 100 100 | 50 50 255 0 101 101 | 0 255 128 128 127 128 | 200 50 128 127 100 101 |

Under a black mask (0) the first frame stays, 200, under a white one (255) the second, 50. In the third pixel
(0 · 127 + 255 · 128 + 127) / 255 = 32 767 / 255 = 128; in the fourth the frames have swapped places, and the result
is 127. The last two pixels show the rounding: the mix of 100 and 101 under the mask 127 is 100.498… and rounds
to 100, while under the mask 128 it is already 100.502… and rounds to 101.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 6000 on `coral-npu`;
- 0 ≤ A[i], B[i], T[i] ≤ 255;
- the answer is compared exactly.
