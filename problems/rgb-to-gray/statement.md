---
code: "17"
title: Shades of Gray
difficulty: easy
tags: [image, bytes]
authors: []
---

Turn a colour picture into a grayscale one: on the GPU (`gpu-cuda`), as a circuit (`rtl`), on a vector RISC-V CPU
(`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

A picture of `N` pixels is given as three planes of bytes: `R`, `G` and `B`. Calculate the brightness of every pixel
with the integer weights of the BT.601 standard:

```
Y[i] = (77 * R[i] + 150 * G[i] + 29 * B[i]) >> 8
```

The weights add up to 256, so the brightness is a byte again: white stays 255.

Implement the function

```cpp
void solve(const uint8_t* R, const uint8_t* G, const uint8_t* B, uint8_t* Y, int N);
```

The planes point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns, the judge synchronizes
afterwards. Every camera pipeline does this to every frame; bytes are where wide vectors and small circuits pay off.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the input `N` and four streams of bytes. The judge raises
`start` for the first cycle and from the next cycle on feeds one pixel per cycle: `R_data`, `G_data` and `B_data`
hold the pixel while `R_valid`, `G_valid` and `B_valid` are high. Hand `Y[i]` back in the same order by raising
`Y_valid` with the byte on `Y_data`, then raise `done`. Registers are zero before `start`. The score is the number of cycles until `done` times the logic
depth (the most gates on a path between registers) times the transistors of the synthesized circuit: smaller is
better. The circuit must be synchronous: no latches, no asynchronous resets.
A multiplication by a constant is a few shifted additions, and a shift is a wire.

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

| R | G | B | Y |
| --- | --- | --- | --- |
| 255 255 0 0 10 | 255 0 255 0 20 | 255 0 0 255 30 | 255 76 149 28 18 |

Pure red: `77 * 255 >> 8 = 76`. The last pixel: `(770 + 3000 + 870) >> 8 = 4640 >> 8 = 18`.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv`,
  1 ≤ N ≤ 6000 on `coral-npu`;
- the planes hold any bytes;
- the answer is compared exactly.
