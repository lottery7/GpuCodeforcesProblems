import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 5},
           "inputs": {"A": [3, -1, 7, 7, 0]},
           "expected": {"R": [7]}}
    # the answer below zero, at the first and at the last place
    yield {"name": "negative", "scalars": {"N": 4}, "inputs": {"A": [-5, -3, -1000000000, -4]}, "measure": False}
    yield {"name": "first", "scalars": {"N": 3}, "inputs": {"A": [1000000000, 5, 999999999]}, "measure": False}
    yield {"name": "last", "scalars": {"N": 3}, "inputs": {"A": [-7, 5, 6]}, "measure": False}
    for seed, n in enumerate([1, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: one array of 24000 bytes fits
    yield {"name": "n-6000", "scalars": {"N": 6000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {"A": torch.randint(-1_000_000_000, 1_000_000_001, (test["N"],), dtype=torch.int64, device=device,
                               generator=gen).to(torch.int32)}


def reference(test, inputs):
    return {"R": inputs["A"].max().reshape(1)}


def validate_input(test, inputs):
    a = inputs["A"]
    return test["N"] >= 1 and bool((a >= -1_000_000_000).all() and (a <= 1_000_000_000).all())
