---
title: Best Stretch
---

Every evening the merchant Afanasy writes the result of the day into his ledger, in kopecks: a profit with a plus,
a loss with a minus. For the fair he wants to boast about his best stretch: several days in a row that brought the
most in total. A stretch cannot be empty: if every day was a loss, he will have to boast about the mildest of them.

You are given an array `A` of `N` integers: `A[i]` is the result of day `i`. Find the largest sum of consecutive
days:

```
R[0] = max(A[l] + A[l+1] + ... + A[r]) over all 0 ≤ l ≤ r < N
```

The sum may not fit 32 bits, so the answer is a 64-bit number.

The problem can be solved on the GPU (`gpu-cuda`), on a simulated GPU (`cuda-sim`), as a circuit (`rtl`), on a vector
RISC-V CPU (`riscv-rvv`) and on the Coral NPU core (`coral-npu`).

Implement the function

```cpp
void solve(const int* A, int64_t* R, int N);
```

`A` and `R` point to GPU memory; `R` holds one 64-bit element. `solve` runs on the CPU: it launches the kernels and
returns, the judge synchronizes afterwards. The usual loop over the days is sequential: every step waits for the
previous one. But pieces of the ledger can be processed independently if a little more is remembered about each
than its best stretch: then the results of two neighbouring pieces combine into the result of their union.

## On the GPU simulator (`cuda-sim`)

The same CUDA `solve`, run in GPGPU-Sim on a model of the NVIDIA Quadro V100 (QV100). The score is the simulated GPU
cycles: smaller is better. `cudaMemcpy*` and warp shuffles (`__shfl_*_sync`) are not allowed in the solution.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve` with the input `N`, an input stream and an output of one element.
The judge raises `start` for the first cycle and from the next cycle on feeds one element per cycle: `A_data` holds
`A[i]` while `A_valid` is high. Hand the answer back by raising `R_valid` for one cycle with the sum on the 64 bits
of `R_data`, then raise `done`. The stream cannot be paused or rewound. Registers are zero before `start`. The score
is the number of cycles until `done` times the logic depth (the most gates on a path between registers) times the
transistors of the synthesized circuit: smaller is better. The circuit must be synchronous: no latches, no
asynchronous resets. How wide the adders have to be follows from the constraints below.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function. The core is 32-bit: a 64-bit number takes two of its registers, and every addition
and comparison of such numbers costs several instructions.

## Example

| A | R |
| --- | --- |
| -200 400 -100 250 -700 300 150 -50 | 550 |

The best stretch is days 1 to 3: 400 − 100 + 250 = 550 kopecks. It pays to sit out the losing day inside it rather
than to cut the stretch there. The loss of 700 kopecks, however, never pays back: there is no point in carrying the
stretch through it, and days 5 and 6 on their own bring only 450.

## Constraints

- 1 ≤ N ≤ 10 000 000 on the GPU, 1 ≤ N ≤ 4096 on `rtl`, 1 ≤ N ≤ 100 000 on `riscv-rvv` and `cuda-sim`,
  1 ≤ N ≤ 6000 on `coral-npu`;
- −1 000 000 ≤ A[i] ≤ 1 000 000;
- the answer is compared exactly.
