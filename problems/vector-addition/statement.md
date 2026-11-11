---
code: "1"
title: Vector Addition
difficulty: easy
tags: [vector, arithmetic]
authors: []
---

Implement vector addition: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a
vector RISC-V CPU (`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given two input vectors `A` and `B` of `N` floats, calculate:

```
C[i] = A[i] + B[i]
```

Implement the function

```cpp
void solve(const float* A, const float* B, float* C, int N);
```

`A`, `B` and `C` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns,
the judge synchronizes afterwards. Your implementation must support all valid values of `N`.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` is not allowed in the solution, the data is already on the GPU.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A`, `B` and `C` are memories of the judge: put an index on
`A_addr` and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`.
Floats travel as their raw IEEE 754 bits, so the circuit has to add binary32 numbers itself. The judge raises
`start` for the first cycle; raise `done` once every element of `C` is written. The score is the number of cycles times
the logic depth (the most gates on a path between registers) times the transistors of the synthesized circuit:
smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; `A`, `B` and `C` are in ordinary
memory. The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
`A`, `B` and `C` lie in that data memory, next to the globals and the stack of the solution. The score is the
instructions the core executed: smaller is better. In C++ declare `N` as `int32_t`, as the solution template does: on
this target `int32_t` is `long`, and a `solve` taking `int` would be another function.

## Example

| A | B | C |
| --- | --- | --- |
| 1 2 3 4 | 5 6 7 8 | 6 8 10 12 |

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv`, 1 ≤ N ≤ 2000 on
  `coral-npu`;
- the inputs are in [0, 1);
- the answer is compared with absolute and relative tolerance `1e-6`.
