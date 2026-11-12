import math

import torch

MAX_SIZE = 10_000_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 20}, "inputs": {},
           "expected": {"C": [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]}}
    for n in [3, 4, 5]:
        yield {"name": f"n-{n}", "scalars": {"N": n}, "measure": False}
    # the last number is the square of a prime: its only divisor to find is the largest one worth trying
    for n in [10, 26, 50, 122, 962]:
        yield {"name": f"square-{n - 1}", "scalars": {"N": n}, "measure": False}
    # ... and here the last number is a prime
    yield {"name": "prime-997", "scalars": {"N": 998}, "measure": False}
    yield {"name": "n-4096", "scalars": {"N": 4096}}
    # the Coral NPU core has 32 KB of data memory for everything: the table of 24000 bytes fits
    yield {"name": "n-24000", "scalars": {"N": 24_000}, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle; 97969 = 313 * 313
    yield {"name": "square-97969", "scalars": {"N": 97_970}, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "processors": ["riscv-rvv", "cuda-sim"]}
    for n in [1_000_000, 10_000_000]:
        yield {"name": f"n-{n}", "scalars": {"N": n}, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {}


def reference(test, inputs):
    n = test["N"]
    prime = torch.ones(n, dtype=torch.uint8)
    prime[:2] = 0
    for p in range(2, math.isqrt(n - 1) + 1):
        if prime[p]:
            prime[p * p::p] = 0
    return {"C": prime}


def validate_input(test, inputs):
    return 3 <= test["N"] <= MAX_SIZE
