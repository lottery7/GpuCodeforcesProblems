import torch

LIMIT = 1_000_000
MAX_SIZE = 10_000_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 8},
           "inputs": {"A": [-200, 400, -100, 250, -700, 300, 150, -50]},
           "expected": {"R": [550]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [-LIMIT]}, "measure": False}
    yield {"name": "n-1-profit", "scalars": {"N": 1}, "inputs": {"A": [5]}, "measure": False}
    # every day is a loss: the stretch is the mildest of them, not an empty one
    yield {"name": "losses", "scalars": {"N": 5}, "inputs": {"A": [-5, -3, -LIMIT, -4, -3]}, "measure": False}
    yield {"name": "zero", "scalars": {"N": 3}, "inputs": {"A": [-2, 0, -3]}, "measure": False}
    yield {"name": "whole", "scalars": {"N": 4}, "inputs": {"A": [1, 2, 3, 4]}, "measure": False}
    # the stretch touches the first or the last day, and a loss inside it is worth sitting out
    yield {"name": "first", "scalars": {"N": 5}, "inputs": {"A": [5, -1, 5, -100, 3]}, "measure": False}
    yield {"name": "last", "scalars": {"N": 5}, "inputs": {"A": [3, -100, 5, -1, 5]}, "measure": False}
    # 4 096 000 000 does not fit 32 bits, signed or not; a running total of the losses does not fit them either
    yield {"name": "n-4096-profit", "scalars": {"N": 4096}, "inputs": {"A": torch.full((4096,), LIMIT, dtype=torch.int32)}}
    yield {"name": "n-4096-loss", "scalars": {"N": 4096}, "inputs": {"A": torch.full((4096,), -LIMIT, dtype=torch.int32)}}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: one array of 24000 bytes fits
    yield {"name": "n-6000", "scalars": {"N": 6000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # Seasons of trade: the book splits into 16 spans, each with its own drift under the daily noise, so the best
    # stretch runs through whole seasons and has to decide which bad ones to sit out.
    n = test["N"]
    drift = torch.randint(-400_000, 400_001, (16,), device=device, generator=gen)
    season = torch.arange(n, device=device) * 16 // n
    noise = torch.randint(-600_000, 600_001, (n,), device=device, generator=gen)
    return {"A": (noise + drift[season]).to(torch.int32)}


def reference(test, inputs):
    # the best stretch ending on day r is the running total after it minus the lowest running total before it
    total = torch.cumsum(inputs["A"].to(torch.int64), dim=0)
    before = torch.cat([torch.zeros(1, dtype=torch.int64, device=total.device), total[:-1]])
    return {"R": (total - torch.cummin(before, dim=0).values).max().reshape(1)}


def validate_input(test, inputs):
    a = inputs["A"]
    return 1 <= test["N"] <= MAX_SIZE and bool((a >= -LIMIT).all() and (a <= LIMIT).all())
