import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"A": [3, 0, 3, 255, 3, 0]}}
    for seed, n in enumerate([1, 1000], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # every byte the same value: the worst case for counters updated in parallel
    yield {"name": "same-byte", "scalars": {"N": 4096}, "inputs": {"A": [7] * 4096}, "measure": False}
    # the Coral NPU core has 32 KB of data memory for everything: 20000 bytes and the 1 KB histogram fit
    yield {"name": "n-20000", "scalars": {"N": 20_000}, "seed": 3, "processors": ["coral-npu", "cuda-sim"]}
    # an emulated CPU counts every instruction
    yield {"name": "n-200000", "scalars": {"N": 200_000}, "seed": 4, "processors": ["riscv-rvv"]}
    for seed, n in enumerate([1_000_000, 50_000_000], start=5):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {"A": torch.randint(0, 256, (test["N"],), dtype=torch.uint8, device=device, generator=gen)}


def reference(test, inputs):
    return {"H": torch.bincount(inputs["A"].to(torch.int64), minlength=256).to(torch.uint32)}


def validate_input(test, inputs):
    return test["N"] >= 1
