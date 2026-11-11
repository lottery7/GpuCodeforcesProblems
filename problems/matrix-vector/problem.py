import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 2, "M": 3},
           "inputs": {"A": [1, 2, 3, 4, 5, 6], "X": [1, 0, -1]},
           "expected": {"Y": [-2, -2]}}
    # products of a million by a million do not fit 32 bits
    yield {"name": "wide", "scalars": {"N": 2, "M": 2},
           "inputs": {"A": [1000000, 1000000, -1000000, 1000000], "X": [1000000, 1000000]}, "measure": False}
    yield {"name": "row", "scalars": {"N": 1, "M": 100}, "seed": 1}
    yield {"name": "column", "scalars": {"N": 100, "M": 1}, "seed": 2}
    yield {"name": "n-32-m-48", "scalars": {"N": 32, "M": 48}, "seed": 3}
    # the Coral NPU core has 32 KB of data memory for everything: a matrix of 24000 bytes fits
    yield {"name": "n-60-m-100", "scalars": {"N": 60, "M": 100}, "seed": 4, "processors": ["coral-npu"]}
    # a simulated GPU counts every cycle
    yield {"name": "n-256-m-200", "scalars": {"N": 256, "M": 200}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "n-512-m-512", "scalars": {"N": 512, "M": 512}, "seed": 6, "processors": ["riscv-rvv"]}
    yield {"name": "n-4096-m-4096", "scalars": {"N": 4096, "M": 4096}, "seed": 7, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n, m = test["N"], test["M"]
    return {"A": torch.randint(-1_000_000, 1_000_001, (n * m,), dtype=torch.int32, device=device, generator=gen),
            "X": torch.randint(-1_000_000, 1_000_001, (m,), dtype=torch.int32, device=device, generator=gen)}


def reference(test, inputs):
    n, m = test["N"], test["M"]
    return {"Y": inputs["A"].to(torch.int64).reshape(n, m) @ inputs["X"].to(torch.int64)}


def validate_input(test, inputs):
    return test["N"] >= 1 and test["M"] >= 1 and all(
        bool((value >= -1_000_000).all() and (value <= 1_000_000).all()) for value in inputs.values())
