---
title: One Generation
---

The computer club runs John Conway's Game of Life on the scoreboard of the school stadium: `N` rows of `M` lamps
each, a lit lamp is a live cell. The scoreboard is large and the spectators are impatient: the next generation has
to be computed as fast as possible.

The field is stored row by row in the array `A`: the cell in row `r` and column `c` is `A[r * M + c]`, 1 is alive
and 0 is dead. The neighbours of a cell are the up to eight cells that touch it by a side or by a corner; there are
no live cells beyond the edge of the field. All cells change at the same time:

- a live cell with 2 or 3 live neighbours stays alive;
- a dead cell with exactly 3 live neighbours comes to life;
- any other cell is dead in the next generation.

Write the next generation to the array `B` in the same form:

```
k = the number of live neighbours of the cell (r, c) in the field A
B[r * M + c] = 1 if k = 3, or if k = 2 and A[r * M + c] = 1; otherwise 0
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint8_t* A, uint8_t* B, int N, int M);
```

`A` and `B` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. Every cell needs nine cells of the old field, and for neighbouring cells these nine are almost the same.
The cost of a solution is in how many times one cell is read and how many checks are spent on the edges of the
field.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.
A cell takes one byte: a vector register holds 32 cells of one row.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` and `M` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve`
taking `int` would be another function.
The two fields of the largest test take 24 000 bytes of the 32 KB: a copy of the field with a border of dead cells
will not fit next to them.

## Example

| N | M | A | B |
| --- | --- | --- | --- |
| 4 | 6 | 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 | 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 |

The same fields row by row, `#` is a live cell:

```
A:  . . # # # .      B:  . . . # . .
    . . . . . .          . . . # . .
    # . . . . .          # # . . . .
    # # . . . .          # # . . . .
```

The three cells in the top row are a "blinker" pressed against the edge of the field. The middle cell survives: it
has two neighbours. The outer ones have one neighbour each and die. A new cell is born below the middle one, and
nothing can be born above it: the field ends there. Each of the three cells in the bottom left corner has two
neighbours, so all of them survive, and the fourth cell of the square is born.

## Constraints

- N, M ≥ 1; N × M ≤ 16 777 216 on the GPU, N × M ≤ 100 000 on `riscv-rvv` and `cuda-sim`, N × M ≤ 12 000 on
  `coral-npu`;
- A[r * M + c] is 0 or 1;
- the answer is compared exactly.
