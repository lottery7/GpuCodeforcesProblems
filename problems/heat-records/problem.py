import torch

LOW, HIGH = -90_000, 60_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"A": [-3500, 1200, 800, 1200, 4100, -200]},
           "expected": {"C": [-3500, 1200, 1200, 1200, 4100, 4100]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [LOW]}}
    # every day is a record, no day after the first is one, and all records are below zero
    yield {"name": "rising", "scalars": {"N": 8}, "inputs": {"A": [-7, -5, -2, 0, 3, 8, 21, HIGH]}, "measure": False}
    yield {"name": "falling", "scalars": {"N": 8}, "inputs": {"A": [HIGH, 21, 8, 3, 0, -2, -5, LOW]}, "measure": False}
    yield {"name": "frost", "scalars": {"N": 5}, "inputs": {"A": [-40000, -41000, -39000, -39500, -39000]}, "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # A slow warming under daily noise: records keep falling to the very end, not only in the first days.
    n = test["N"]
    trend = torch.linspace(-30_000, 20_000, n, device=device)
    noise = torch.randint(-25_000, 25_001, (n,), device=device, generator=gen)
    return {"A": (trend.to(torch.int64) + noise).clamp(LOW, HIGH).to(torch.int32)}


def reference(test, inputs):
    return {"C": torch.cummax(inputs["A"], dim=0).values}


def validate_input(test, inputs):
    a = inputs["A"]
    return test["N"] >= 1 and bool((a >= LOW).all() and (a <= HIGH).all())
