---
title: Three-Point Stencil
---

Add every element to its two neighbours: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector RISC-V CPU
(`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given `N` integers `A`, calculate

```
C[i] = A[i-1] + A[i] + A[i+1]
```

where the neighbours that do not exist, `A[-1]` and `A[N]`, count as zero.

Implement the function

```cpp
void solve(const int* A, int* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns, the judge synchronizes
afterwards. A blur, a heat equation and a finite difference are all this loop; the edges are where the bugs live.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the input `N` and two streams. The judge raises `start` for
the first cycle and from the next cycle on feeds one element per cycle: `A_data` holds `A[i]` while `A_valid` is
high. Hand `C[i]` back in the same order by raising `C_valid` with the element on `C_data`, then raise `done`. The
stream cannot be paused or rewound: `C[i]` needs `A[i+1]`, so the circuit has to remember what it has seen.
Registers are zero before `start`. The score is the number of cycles until `done` times the logic
depth (the most gates on a path between registers) times the transistors of the synthesized circuit: smaller is
better. The circuit must be synchronous: no latches, no asynchronous resets.

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
| 1 2 3 4 5 | 3 6 9 12 9 |

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- 0 ≤ A[i] ≤ 1 000 000;
- the answer is compared exactly.
