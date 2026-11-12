---
title: Prefix Sum
---

Build a circuit that computes prefix sums of a stream of numbers.

Given `N` 32-bit signed integers `A`, compute

```
C[i] = A[0] + A[1] + ... + A[i]
```

with 32-bit wrap-around, as an adder of 32 bits does.

The submission is a SystemVerilog module `solve`. The judge raises `start` for the first cycle and from the
next cycle on feeds one element of `A` per cycle: `A_data` holds `A[0]`, `A[1]`, ... while `A_valid` is high. The
circuit hands `C` back the same way: on every rising edge of `clk` where `C_valid` is high the judge takes `C_data`
as the next element of `C`. Raise `done` once all `N` elements of `C` are handed over. Registers are zero before
`start`.

The score is the number of cycles until `done` times the logic depth (the most gates on a path between registers) times the transistors of the synthesized circuit:
smaller is better. The circuit must be synchronous: no latches, no asynchronous resets.

## Example

| A | C |
| --- | --- |
| 1 2 3 4 −10 | 1 3 6 10 0 |

## Constraints

- 1 ≤ N ≤ 100 000;
- the answer is compared exactly.
