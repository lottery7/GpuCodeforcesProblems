---
title: Encoder Disc
---

The dish of a deep space communication station is turned by a shaft, and on the shaft sits the disc of an angle
sensor with 32 tracks. The tracks are laid out in Gray code: the codes of neighbouring positions differ in exactly
one bit, so on the border of two positions the sensor cannot read anything but one of the two. Position number `x`
is written on the disc as `x ^ (x >> 1)`, where `^` is the bitwise exclusive OR, while the pointing system needs the
numbers themselves.

You are given an array `A` of `N` sensor readings: `A[i]` is the Gray code of some position. For every reading find
the number of the position `C[i]`, that is, the number such that

```
A[i] = C[i] ^ (C[i] >> 1)
```

For any 32-bit code such a number exists and is unique.

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint32_t* A, uint32_t* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. Bit `k` of the number is the XOR of all bits of the code from bit `k` up to the top one. Walking the bits
one by one is 31 steps per reading; the same answer can be put together in just a few.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A` and `C` are memories of the judge: put an index on `A_addr`
and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`. The judge
raises `start` for the first cycle; raise `done` once every element of `C` is written. Registers are zero before
`start`. The score is the number of cycles until `done` times the logic depth (the most gates on a path between
registers) times the transistors of the synthesized circuit: smaller is better. The circuit must be synchronous: no
latches, no asynchronous resets. A chain of 31 XOR gates is exactly that logic depth: it can be cut several times
over, but the price is transistors, and the address counter has a depth of its own.

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
| 0 1 3 2 6 7 5 4 12 2147483648 | 0 1 2 3 4 5 6 7 8 4294967295 |

The first eight readings are the codes of positions 0 to 7 in a row: each differs from the one before in a single
bit. The code 12 = `1100₂` belongs to position 8 = `1000₂`, because `1000₂ ^ 0100₂ = 1100₂`. The last reading has
only the top bit set, and that bit reaches every bit of the answer: it is position 4 294 967 295.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- 0 ≤ A[i] ≤ 4 294 967 295;
- the answer is compared exactly.
