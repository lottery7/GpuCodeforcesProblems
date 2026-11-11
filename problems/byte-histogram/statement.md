---
code: "6"
title: Byte Histogram
difficulty: medium
tags: [histogram, bytes]
authors: []
---

Count how many times every byte value occurs: on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), on a vector
RISC-V CPU (`riscv-rvv`) or on the Coral NPU core (`coral-npu`).

Given `N` bytes `A`, fill the 256 counters

```
H[v] = the number of i with A[i] == v,    v = 0 ... 255
```

Implement the function

```cpp
void solve(const uint8_t* A, uint32_t* H, int N);
```

`A` and `H` point to GPU memory. `solve` runs on the CPU: it launches the kernels and returns, the judge synchronizes
afterwards. `H` holds garbage before the call: every counter has to be written, the zero ones too.

Unlike a sum, a histogram writes to a place that depends on the data: threads of a GPU collide on a counter, and
lanes of a vector cannot update the same counter at once.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Tesla V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory, next to the globals and the stack of the solution. The score is the instructions
the core executed: smaller is better. In C++ declare `N` as `int32_t`, as the solution template does: on this target
`int32_t` is `long`, and a `solve` taking `int` would be another function.

## Example

| A | H |
| --- | --- |
| 3 0 3 255 3 0 | H[0] = 2, H[3] = 3, H[255] = 1, the other 253 counters are 0 |

## Constraints

- 1 ≤ N ≤ 50 000 000 on the GPU, 1 ≤ N ≤ 200 000 on `riscv-rvv`, 1 ≤ N ≤ 20 000 on `cuda-sim` and `coral-npu`;
- the bytes take any value, all of them may be equal;
- the answer is compared exactly.
