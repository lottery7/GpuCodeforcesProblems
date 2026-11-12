---
title: Collatz Steps
---

Count the steps of the Collatz walk for every number: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector RISC-V CPU
(`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

One step replaces `x` with `x / 2` if `x` is even and with `3 * x + 1` if it is odd. Nobody has proved that every
walk reaches 1, but for the numbers of this problem it does. Given `N` positive integers `A`, calculate

```
C[i] = the number of steps from A[i] to 1
```

Implement the function

```cpp
void solve(const uint32_t* A, int* C, int N);
```

`A` and `C` point to GPU memory. `solve` runs on the CPU: it launches a kernel and returns, the judge synchronizes
afterwards. The walks have different lengths, from 0 to 524 steps: threads of one warp and lanes of one vector wait
for the longest of them. On the way the numbers grow far above the start, up to 56 991 483 520, which does not fit
32 bits.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A` and `C` are memories of the judge: put an index on `A_addr`
and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`. The judge
raises `start` for the first cycle; raise `done` once every element of `C` is written. Registers are zero before `start`. The score is the number of cycles until `done` times the logic
depth (the most gates on a path between registers) times the transistors of the synthesized circuit: smaller is
better. The circuit must be synchronous: no latches, no asynchronous resets.
Halving is a wire, `3 * x + 1` is one adder, and nothing says a cycle has to be one step.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.
A 64-bit number takes two registers of this 32-bit core.

## Example

| A | C |
| --- | --- |
| 1 2 3 6 7 27 | 0 1 7 8 16 111 |

`6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1` is 8 steps.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 1024 on `rtl`, 1 ≤ N ≤ 50 000 on `riscv-rvv`, 1 ≤ N ≤ 4096 on `cuda-sim`,
  1 ≤ N ≤ 2000 on `coral-npu`;
- 1 ≤ A[i] ≤ 1 000 000;
- the answer is compared exactly.
