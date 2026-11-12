---
title: Map Tiles
---

A map service cuts the map of the world into tiles: 65 536 columns and 65 536 rows. All the tiles lie in one file
in the order of the Z-curve: the number of a tile in the file is what you get by shuffling the bits of its column
and of its row like the two halves of a deck, one from each in turn. In this order tiles that are neighbours on the
map usually end up close in the file too, and a screen of the map is read from the disk almost in one piece.

You are given arrays `X` and `Y` of `N` integers: `X[i]` is the column of a tile and `Y[i]` is its row. For every
tile find its number in the file `C[i]`. Bits are numbered from zero, starting from the lowest:

```
bit 2k of C[i]     = bit k of X[i]
bit 2k + 1 of C[i] = bit k of Y[i]        (k = 0, 1, ..., 15)
```

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const uint32_t* X, const uint32_t* Y, uint32_t* C, int N);
```

`X`, `Y` and `C` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge
synchronizes afterwards. Moving the bits one at a time is 32 steps per tile; they can be spread apart in whole
groups as well. And in a circuit a permutation of bits costs nothing at all.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `X`, `Y` and `C` are memories of the judge: put an index on
`X_addr` and the element arrives on `X_data` on the next cycle (`Y_addr` and `Y_data` work the same way); write
`C[C_addr] = C_data` by raising `C_write`. The judge raises `start` for the first cycle; raise `done` once every
element of `C` is written. Registers are zero before `start`. The score is the number of cycles until `done` times
the logic depth (the most gates on a path between registers) times the transistors of the synthesized circuit:
smaller is better. The circuit must be synchronous: no latches, no asynchronous resets. Shuffling the bits is just
connecting wires, so all the logic of the circuit goes into walking the array: that is what the score will measure.

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

| X | Y | C |
| --- | --- | --- |
| 0 1 0 3 65535 65535 | 0 0 1 5 0 65535 | 0 1 2 39 1431655765 4294967295 |

For the tile in column 3 = `011₂` and row 5 = `101₂` the bits of the row `1 0 1` go to the odd places and the bits
of the column `0 1 1` to the even ones: the result is `100111₂` = 39. The tile in column 65 535 and row 0 has every
even bit of its number set, which is 1 431 655 765, and the last tile of the map has all 32.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 2000 on `coral-npu`;
- 0 ≤ X[i], Y[i] ≤ 65 535;
- the answer is compared exactly.
