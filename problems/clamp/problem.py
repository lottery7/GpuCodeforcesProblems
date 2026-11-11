import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6, "LO": -2, "HI": 5},
           "inputs": {"A": [-7, -2, 0, 5, 9, 3]},
           "expected": {"C": [-2, -2, 0, 5, 5, 3]}}
    yield {"name": "n-1", "scalars": {"N": 1, "LO": 0, "HI": 0}, "seed": 1}
    # a range of one value, and a range nothing falls out of
    yield {"name": "point", "scalars": {"N": 100, "LO": 17, "HI": 17}, "seed": 2, "measure": False}
    yield {"name": "wide", "scalars": {"N": 100, "LO": -1_000_000, "HI": 1_000_000}, "seed": 3, "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024, "LO": -300_000, "HI": 250_000}, "seed": 4}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096, "LO": -1000, "HI": 500_000}, "seed": 5, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000, "LO": -500_000, "HI": 500_000}, "seed": 6, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000, "LO": -400_000, "HI": 600_000}, "seed": 7,
           "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n, "LO": -123_456, "HI": 654_321}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {"A": torch.randint(-1_000_000, 1_000_001, (test["N"],), dtype=torch.int32, device=device, generator=gen)}


def reference(test, inputs):
    return {"C": inputs["A"].clamp(test["LO"], test["HI"])}


def validate_input(test, inputs):
    a = inputs["A"]
    return (test["N"] >= 1 and -1_000_000 <= test["LO"] <= test["HI"] <= 1_000_000
            and bool((a >= -1_000_000).all() and (a <= 1_000_000).all()))
