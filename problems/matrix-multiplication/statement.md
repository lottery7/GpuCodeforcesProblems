---
code: "2"
title: Matrix Multiplication
difficulty: medium
tags: [matrix, arithmetic]
authors: []
---

Implement matrix multiplication: on the GPU (`gpu-cuda`), as a circuit (`rtl`), on a vector RISC-V CPU (`riscv-rvv`)
or on the Coral NPU core (`coral-npu`).

Given two dense matrices `A` (M×K) and `B` (K×N), stored row by row, compute the product
`C = A × B` (M×N, row by row).

```cpp
void solve(const float* A, const float* B, float* C, int M, int K, int N);
```

`A`, `B` and `C` point to GPU memory. `solve` runs on the CPU and launches kernels.
Your implementation must handle the tiling and memory access patterns efficiently.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A`, `B` and `C` are memories of the judge: put an element index
on `A_addr` and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`.
Floats travel as their raw IEEE 754 bits, so the circuit multiplies and adds binary32 numbers itself. The judge
raises `start` for the first cycle; raise `done` once all of `C` is written. The score is the number of cycles times
the logic depth (the most gates on a path between registers) times the transistors of the synthesized circuit:
smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; `A`, `B` and `C` are in ordinary
memory. The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
`A`, `B` and `C` lie in that data memory, next to the globals and the stack of the solution, so there is little room
for copies of the matrices. The score is the instructions the core executed: smaller is better. In C++ declare the
sizes as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking `int`
would be another function.

## Example

`M = 2, K = 3, N = 2`, `A = [[1, 2, 3], [4, 5, 6]]`, `B = [[1, 0], [0, 1], [1, 1]]`,
answer `C = [[4, 5], [10, 11]]`.

## Constraints

- 1 ≤ M, K, N ≤ 1000 on the GPU, 1 ≤ M, K, N ≤ 64 on `rtl`, 1 ≤ M, K, N ≤ 128 on `riscv-rvv`, 1 ≤ M, K, N ≤ 48 on
  `coral-npu`;
- the elements of `A` and `B` are in [0, 1);
- the answer is compared with absolute and relative tolerance `1e-3`.
