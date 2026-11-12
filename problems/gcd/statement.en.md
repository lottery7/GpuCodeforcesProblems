---
title: Greatest Common Divisor
---

Compute greatest common divisors: as a circuit (`rtl`), on a vector RISC-V CPU (`riscv-rvv`) or on the Coral NPU
core (`coral-npu`).

Given two arrays `A` and `B` of `N` positive integers, calculate

```
C[i] = gcd(A[i], B[i])
```

Implement the function

```cpp
void solve(const uint32_t* A, const uint32_t* B, uint32_t* C, int32_t N);
```

The algorithm is twenty-three centuries old; what it costs depends on what the processor can do in one step. A
remainder is one instruction on a CPU and a large slow block in a circuit, while a shift is free in a circuit and a
comparison is cheap.

## As a circuit (`rtl`)

The submission is a SystemVerilog module `solve`. `A`, `B` and `C` are memories of the judge: put an index on
`A_addr` and the element arrives on `A_data` on the next cycle; write `C[C_addr] = C_data` by raising `C_write`.
The judge raises `start` for the first cycle; raise `done` once every element of `C` is written. Registers are zero
before `start`. The score is the number of cycles times the logic depth (the most gates on a path between registers)
times the transistors of the synthesized circuit: smaller is better. The tests hold pairs such as `(1, 1000000000)`:
a circuit that only subtracts the smaller number from the larger would spend a billion cycles on one of them. The
circuit must be synchronous: no latches, no asynchronous resets.

## On RISC-V (`riscv-rvv`)

The same `solve` in C, C++ or RISC-V assembly for RV64GCV with 256-bit vectors; the arrays are in ordinary memory.
The score is the instructions executed plus 10 per L1 miss and 100 per L2 miss: smaller is better. A division is one
instruction here, as cheap as an addition.

## On the Coral NPU (`coral-npu`)

The same `solve` in C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x,
Zve32f), no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data;
the arrays lie in that data memory. The score is the instructions the core executed: smaller is better. In C++
declare `N` as `int32_t`, as the solution template does: on this target `int32_t` is `long`, and a `solve` taking
`int` would be another function.

## Example

| A | B | C |
| --- | --- | --- |
| 12 17 1000000000 7 | 18 5 400000000 7 | 6 1 200000000 7 |

## Constraints

- 1 ≤ N ≤ 256 on `rtl`, 1 ≤ N ≤ 50 000 on `riscv-rvv`, 1 ≤ N ≤ 2000 on `coral-npu`;
- 1 ≤ A[i], B[i] ≤ 1 000 000 000;
- the answer is compared exactly.
