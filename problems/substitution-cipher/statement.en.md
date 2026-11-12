---
title: Substitution Table
---

Before going on the air, the radio operator of the icebreaker Taimyr enciphers every radiogram with a substitution
table: instead of a byte with the value `v`, the byte `T[v]` is transmitted. The captain hands out a new table every
day, and a watch brings so many radiograms that the operator has long been tired of rewriting them by hand.

You are given a table `T` of 256 bytes and a message `A` of `N` bytes. Encipher the message:

```
C[i] = T[A[i]]
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint8_t* T, const uint8_t* A, uint8_t* C, int N);
```

`T`, `A` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. There is nothing to compute here: every byte of the message is an address to look up in the
table. The whole cost of a solution is in how the processor reads memory at an address it does not know in advance.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `T`, `A` and `C` are memories of the judge: put an index on
`A_addr` and the element arrives on `A_data` on the next cycle (`T` is read the same way, through `T_addr` and
`T_data`); `C[C_addr] = C_data` is written on a cycle when `C_write` is high. The judge raises `start` for the first
cycle; raise `done` once every element of `C` is written. Registers are zero before `start`. The score is the number
of cycles until `done` times the logic depth (the most gates on a path between registers) times the transistors of
the synthesized circuit: smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.
The two reads are chained: the byte that arrives from `A` is the address for `T`, so one element is on its way for
several cycles. But the next address may be issued without waiting for the previous element to reach the end.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.
The vector extension has an indexed load: it takes its addresses from another vector.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.
The table, the message and the answer share one data memory: in the largest test they take 24 256 bytes of the 32 KB.

## Example

| T | A | C |
| --- | --- | --- |
| T[v] = (7 · v + 3) mod 256 | 72 105 33 200 0 255 | 251 226 234 123 3 252 |

The byte 72 becomes `T[72]` = (7 · 72 + 3) mod 256 = 507 mod 256 = 251. A zero byte is enciphered too: `T[0]` = 3.
The bytes 200 and 255 are above 127, yet they index the table like any other: `T[200]` = 123, `T[255]` = 252.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 12 000 on `coral-npu`;
- the table `T` has exactly 256 bytes; `T[v]` and `A[i]` take any value from 0 to 255;
- the answer is compared exactly.
