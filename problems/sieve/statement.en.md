---
title: Table of Primes
---

At the cipher office keys are put together from prime numbers, and before every shift a fresh table goes up on the
wall: next to every number from 0 to N − 1 there is a one if the number is prime and a zero if it is not. Petya the
trainee made such a table by honestly trying every divisor of every number in turn, and did not finish before the
shift began. Tomorrow the table is yours to make.

You are given a number `N`. Fill the array `C` of `N` bytes:

```
C[i] = 1 if i is prime, and 0 otherwise
```

A prime is an integer greater than 1 that is divisible only by 1 and by itself; 0 and 1 are not primes. The problem
has no input arrays.

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector RISC-V CPU
(`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(uint8_t* C, int N);
```

`C` points to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. Trying every divisor of every number takes too long: the limits are set for the sieve of Eratosthenes.
But the sieve takes the primes one after another and walks the whole table for each: how to hand such work out to
threads, vectors and the cache has to be decided anew on every processor.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution. The
cycles of the simulator are time, not the amount of work: the threads compute at the same time, but a long chain
of divisions or memory accesses in one thread is expensive. The cheapest solution here may be built differently
from the one for a real GPU.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better. A table of
100 000 bytes is larger than the L1 cache (32 KiB): a sieve that walks all of it for every prime pays in misses.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function. The table takes most of the data memory: there is no room for a second one like it.

## Example

| N | C |
| --- | --- |
| 20 | 0 0 1 1 0 1 0 1 0 0 0 1 0 1 0 0 0 1 0 1 |

The primes below 20 are 2, 3, 5, 7, 11, 13, 17 and 19: the table has ones at these places.

## Constraints

- 3 ≤ N ≤ 10 000 000 on the GPU, 3 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`, 3 ≤ N ≤ 24 000 on `coral-npu`;
- the answer is compared exactly.
