import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 4},
           "inputs": {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]},
           "expected": {"R": [70]}}
    for seed, n in enumerate([1, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 8000 bytes fit
    yield {"name": "n-2000", "scalars": {"N": 2000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=6):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n = test["N"]
    return {
        "A": torch.randint(0, 1001, (n,), dtype=torch.int32, device=device, generator=gen),
        "B": torch.randint(0, 1001, (n,), dtype=torch.int32, device=device, generator=gen),
    }


def reference(test, inputs):
    return {"R": (inputs["A"].to(torch.int64) * inputs["B"].to(torch.int64)).sum().reshape(1)}


def validate_input(test, inputs):
    return test["N"] >= 1 and all(bool((value >= 0).all() and (value <= 1000).all()) for value in inputs.values())
