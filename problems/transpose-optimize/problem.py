import torch


def tests():
    yield {"name": "sample", "sample": True, "measure": False, "scalars": {"N": 2, "M": 3},
           "inputs": {"A": [1, 2, 3, 4, 5, 6]},
           "expected": {"B": [1, 4, 2, 5, 3, 6]}}
    for seed, (n, m) in enumerate([(1, 37), (41, 1)], start=1):
        yield {"name": f"{n}x{m}", "measure": False, "scalars": {"N": n, "M": m}, "seed": seed}
    # neither matrix fits the 1 MiB L2 of the cache model: the order of the walk decides the score
    for seed, (n, m) in enumerate([(1024, 1024), (700, 1000)], start=3):
        yield {"name": f"{n}x{m}", "scalars": {"N": n, "M": m}, "seed": seed}


def generate(test, gen, device):
    return {"A": torch.rand(test["N"] * test["M"], generator=gen, device=device)}


def reference(test, inputs):
    return {"B": inputs["A"].view(test["N"], test["M"]).t().contiguous().reshape(-1)}
