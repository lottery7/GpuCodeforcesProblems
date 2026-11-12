---
title: One Instruction or Thirty-Two Lanes
---

Both codes count the bits in which two arrays of 32-bit words differ, on the Coral NPU core (`coral-npu`): RV32IMF
with Zbb and 128-bit vectors. The first relies on `cpop`, the Zbb instruction that counts the bits of a word. The
second never uses it: it counts bits with shifts, masks and a multiplication, in vector registers. Which one costs
less?

The score is the instructions the core executed.
