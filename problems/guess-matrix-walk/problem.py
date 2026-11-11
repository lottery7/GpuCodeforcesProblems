import torch


def tests():
    yield {"name": "sample", "sample": True, "measure": False, "scalars": {"N": 2},
           "inputs": {"A": [1, 2, 3, 4]},
           "expected": {"R": [10]}}
    # a matrix of 768 x 768 or 1024 x 1024 ints does not fit the 1 MiB L2 of the cache model
    for seed, n in enumerate([768, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}


def generate(test, gen, device):
    n = test["N"]
    return {"A": torch.randint(-1000, 1001, (n * n,), dtype=torch.int32, device=device, generator=gen)}


def reference(test, inputs):
    return {"R": inputs["A"].to(torch.int64).sum().reshape(1)}
