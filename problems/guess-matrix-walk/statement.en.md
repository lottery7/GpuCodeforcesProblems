---
title: Rows or Columns
---

Both codes add up all the elements of a square matrix on a vector RISC-V CPU (`riscv-rvv`). The matrix of `N` rows
and `N` columns is stored row by row, and the two loops differ only in the order of the walk. Which one costs less?

The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss; the cache model has a 32 KiB L1 and
a 1 MiB L2 with 64-byte lines. Both codes are compiled with the same optimizing compiler.
