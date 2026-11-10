import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"M": 2, "K": 3, "N": 2},
           "inputs": {"A": [1, 2, 3, 4, 5, 6], "B": [1, 0, 0, 1, 1, 1]},
           "expected": {"C": [4, 5, 10, 11]}}
    sizes = [(1, 1, 1), (32, 32, 32), (256, 256, 256), (1000, 333, 777), (1000, 1000, 1000)]
    for seed, (m, k, n) in enumerate(sizes, start=1):
        yield {"name": f"{m}x{k}x{n}", "scalars": {"M": m, "K": k, "N": n}, "seed": seed}


def generate(test, gen, device):
    m, k, n = test["M"], test["K"], test["N"]
    return {
        "A": torch.rand(m * k, generator=gen, device=device),
        "B": torch.rand(k * n, generator=gen, device=device),
    }


def reference(test, inputs):
    m, k, n = test["M"], test["K"], test["N"]
    # fp64 accumulation for the reference, stored as fp32
    a = inputs["A"].view(m, k).double()
    b = inputs["B"].view(k, n).double()
    return {"C": (a @ b).float().reshape(-1)}
