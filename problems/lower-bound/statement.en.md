---
title: Tournament Table
---

In the final of the city quiz a right answer earns a team points and a wrong one takes points away, so a total can
end up below zero. The totals of all the teams are already written into a table in non-decreasing order. Now the
fans are calling the studio, and they all ask the same thing, each with a number of their own: "How many teams
scored fewer points than this?"

You are given an array `A` of `N` integers in non-decreasing order: `A[i]` is the total of team `i`. You are also
given an array `Q` of `M` questions. For every question find

```
C[j] = the number of indices i with A[i] < Q[j]
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const int* A, const int* Q, int* C, int N, int M);
```

`A`, `Q` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. The questions do not depend on each other, but inside one question every read of the table
depends on the previous one: a cheap solution runs many searches at once and wastes nothing on a step.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the inputs `N` and `M`. `A`, `Q` and `C` are memories of the
judge: put an index on `A_addr` and the element arrives on `A_data` on the next cycle (`Q_addr` and `Q_data` work the
same way); write `C[C_addr] = C_data` by raising `C_write`. The judge raises `start` for the first cycle; raise
`done` once every element of `C` is written. Registers are zero before `start`. The score is the number of cycles
until `done` times the logic depth (the most gates on a path between registers) times the transistors of the
synthesized circuit: smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.
The memory hands over one number of `A` per cycle, so a question costs the circuit about log N cycles; a read past
the end of an array returns 0.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better. A table of
100 000 numbers does not fit the L1 cache (32 KiB): it matters not only how many times a search reads the table,
but also where.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` and `M` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve`
taking `int` would be another function.

## Example

| A | Q | C |
| --- | --- | --- |
| -20 -5 -5 0 10 10 10 35 | 10 -30 36 0 -5 11 35 | 4 0 8 3 1 7 7 |

Four teams scored fewer than 10 points: the three teams with exactly 10 do not count. Nobody scored fewer than −30,
and all eight scored fewer than 36. The answer to "fewer than −5" is 1: neither of the two teams with a total of −5
is included.

## Constraints

- 1 ≤ N, M ≤ 10 000 000 on the GPU; 1 ≤ N ≤ 4096 and 1 ≤ M ≤ 1024 on `rtl`; 1 ≤ N, M ≤ 100 000 on `riscv-rvv`
  and `cuda-sim`; 1 ≤ N ≤ 3000 and 1 ≤ M ≤ 1500 on `coral-npu`;
- −1 000 000 000 ≤ A[i], Q[j] ≤ 1 000 000 000;
- A[0] ≤ A[1] ≤ ... ≤ A[N−1];
- the answer is compared exactly.
