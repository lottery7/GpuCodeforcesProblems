import torch


def tests():
    yield {"name": "sample", "sample": True, "measure": False, "scalars": {"N": 3},
           "inputs": {"A": [0, 255, 5], "B": [1, 15, 5]},
           "expected": {"R": [5]}}
    yield {"name": "n-1", "measure": False, "scalars": {"N": 1}, "seed": 1}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    for seed, n in enumerate([1024, 3000], start=2):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}


def generate(test, gen, device):
    n = test["N"]
    # torch draws no uint32 directly: draw the 32 bits in 64 and narrow them
    return {name: torch.randint(0, 2**32, (n,), dtype=torch.int64, device=device, generator=gen).to(torch.uint32)
            for name in ("A", "B")}


def reference(test, inputs):
    diff = (inputs["A"].to(torch.int64) ^ inputs["B"].to(torch.int64)) & 0xFFFFFFFF
    count = torch.zeros_like(diff)
    for bit in range(32):
        count += (diff >> bit) & 1
    return {"R": count.sum().reshape(1).to(torch.uint32)}
