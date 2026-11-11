import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 4, "K": 3},
           "inputs": {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]},
           "expected": {"C": [13, 26, 39, 52]}}
    yield {"name": "n-1", "scalars": {"N": 1, "K": 7}, "seed": 1}
    yield {"name": "k-0", "scalars": {"N": 100, "K": 0}, "seed": 2, "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024, "K": 1000}, "seed": 3}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096, "K": 999}, "seed": 4, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: three arrays of 8000 bytes fit
    yield {"name": "n-2000", "scalars": {"N": 2000, "K": 123}, "seed": 5, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000, "K": 500}, "seed": 6, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n, "K": 777}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n = test["N"]
    return {name: torch.randint(0, 1001, (n,), dtype=torch.int32, device=device, generator=gen) for name in ("A", "B")}


def reference(test, inputs):
    return {"C": (test["K"] * inputs["A"] + inputs["B"]).to(torch.int32)}


def validate_input(test, inputs):
    return test["N"] >= 1 and 0 <= test["K"] <= 1000 and all(
        bool((value >= 0).all() and (value <= 1000).all()) for value in inputs.values())
