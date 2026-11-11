import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 5},
           "inputs": {"A": [1, 2, 3, 4, -10]},
           "expected": {"C": [1, 3, 6, 10, 0]}}
    for seed, n in enumerate([1, 1000, 4096], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # correctness only: the answer is checked, the cycles neither count nor are limited
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 4, "measure": False}


def generate(test, gen, device):
    n = test["N"]
    return {"A": torch.randint(-2**31, 2**31, (n,), dtype=torch.int32, device=device, generator=gen)}


def reference(test, inputs):
    # 32-bit sums wrap around, like the adder of a circuit
    total = torch.cumsum(inputs["A"].to(torch.int64), dim=0)
    return {"C": ((total + 2**31) % 2**32 - 2**31).to(torch.int32)}
