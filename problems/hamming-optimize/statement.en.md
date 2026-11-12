---
title: Speed Up the Bit Count
---

The editor holds a working Hamming distance for the Coral NPU core (`coral-npu`): it looks at the differing bits one
at a time. Make it cheaper.

`A` and `B` are arrays of `N` 32-bit words. `R[0]` receives the number of bit positions in which they differ:

```
R[0] = popcount(A[0] ^ B[0]) + ... + popcount(A[N - 1] ^ B[N - 1])
```

```cpp
void solve(const uint32_t* A, const uint32_t* B, uint32_t* R, int32_t N);
```

The solution is C or C++ for the Coral NPU core, bare metal: RV32IMF with Zbb and 128-bit vectors (Zve32x, Zve32f),
no operating system and no C++ standard library. The core has 8 KB of memory for code and 32 KB for data; the arrays
lie in that data memory. The score is the instructions the core executed, shown next to the score of the original
code: smaller is better. Zbb counts the bits of a word in one instruction - and that is not the end of the road.
In C++ declare `N` as `int32_t`: on this target `int32_t` is `long`, and a `solve` taking `int` would be another
function.

## Example

| A | B | R |
| --- | --- | --- |
| 0 255 5 | 1 15 5 | 5 |

## Constraints

- 1 ≤ N ≤ 3000;
- the words take any 32-bit value;
- the answer is compared exactly.
