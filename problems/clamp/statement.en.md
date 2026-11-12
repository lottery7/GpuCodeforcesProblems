---
title: Clamp
---

Bring every element of a vector into a range: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector RISC-V CPU
(`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given a vector `A` of `N` integers and two bounds `LO` ≤ `HI`, calculate

```
C[i] = min(max(A[i], LO), HI)
```

Implement the function

```cpp
void solve(const int* A, int* C, int N, int LO, int HI);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns, the judge synchronizes
afterwards. ReLU and saturation are the same operation with other bounds; a branch per element is the one thing no
processor here likes.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the inputs `LO`, `HI` and `N` and two streams. The judge raises
`start` for the first cycle and from the next cycle on feeds one element per cycle: `A_data` holds `A[i]` while
`A_valid` is high. Hand `C[i]` back in the same order by raising `C_valid` with the element on `C_data`, then raise
`done`. Registers are zero before `start`. The score is the number of cycles until `done` times the logic
depth (the most gates on a path between registers) times the transistors of the synthesized circuit: smaller is
better. The circuit must be synchronous: no latches, no asynchronous resets.
Two comparators and a multiplexer: their width is what the constraints leave to you.

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

| LO | HI | A | C |
| --- | --- | --- | --- |
| -2 | 5 | -7 -2 0 5 9 3 | -2 -2 0 5 5 3 |

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 3000 on `coral-npu`;
- -1 000 000 ≤ A[i] ≤ 1 000 000 and -1 000 000 ≤ LO ≤ HI ≤ 1 000 000;
- the answer is compared exactly.
