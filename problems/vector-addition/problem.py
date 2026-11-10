import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 4},
           "inputs": {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]},
           "expected": {"C": [6.0, 8.0, 10.0, 12.0]}}
    for seed, n in enumerate([1, 1024, 1_000_000, 10_000_000], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}


def generate(test, gen, device):
    n = test["N"]
    return {
        "A": torch.rand(n, generator=gen, device=device),
        "B": torch.rand(n, generator=gen, device=device),
    }


def reference(test, inputs):
    return {"C": inputs["A"] + inputs["B"]}
