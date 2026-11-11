import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 5},
           "inputs": {"R": [255, 255, 0, 0, 10], "G": [255, 0, 255, 0, 20], "B": [255, 0, 0, 255, 30]},
           "expected": {"Y": [255, 76, 149, 28, 18]}}
    yield {"name": "black", "scalars": {"N": 3}, "inputs": {"R": [0, 1, 0], "G": [0, 1, 1], "B": [0, 1, 0]}, "measure": False}
    for seed, n in enumerate([1, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit is simulated cycle by cycle: a 64x64 picture for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: four planes of 6000 bytes fit
    yield {"name": "n-6000", "scalars": {"N": 6000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n = test["N"]
    return {name: torch.randint(0, 256, (n,), dtype=torch.uint8, device=device, generator=gen) for name in ("R", "G", "B")}


def reference(test, inputs):
    r, g, b = (inputs[name].to(torch.int32) for name in ("R", "G", "B"))
    return {"Y": ((77 * r + 150 * g + 29 * b) >> 8).to(torch.uint8)}


def validate_input(test, inputs):
    return test["N"] >= 1
