import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 5},
           "inputs": {"A": [1, 2, 3, 4, 5]},
           "expected": {"C": [3, 6, 9, 12, 9]}}
    # the shortest arrays: every element is an edge
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [7]}}
    yield {"name": "n-2", "scalars": {"N": 2}, "inputs": {"A": [1000000, 1000000]}, "measure": False}
    yield {"name": "n-3", "scalars": {"N": 3}, "inputs": {"A": [1000000, 1000000, 1000000]}, "measure": False}
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
    return {"A": torch.randint(0, 1_000_001, (test["N"],), dtype=torch.int32, device=device, generator=gen)}


def reference(test, inputs):
    padded = torch.nn.functional.pad(inputs["A"], (1, 1))
    return {"C": padded[:-2] + padded[1:-1] + padded[2:]}


def validate_input(test, inputs):
    a = inputs["A"]
    return test["N"] >= 1 and bool((a >= 0).all() and (a <= 1_000_000).all())
