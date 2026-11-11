import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 4},
           "inputs": {"A": [12, 17, 1000000000, 7], "B": [18, 5, 400000000, 7]},
           "expected": {"C": [6, 1, 200000000, 7]}}
    # the cases a loop of subtractions would spend a billion steps on, and the longest run of the Euclidean algorithm
    yield {"name": "extremes", "scalars": {"N": 6},
           "inputs": {"A": [1, 1000000000, 999999999, 701408733, 536870912, 999999937],
                      "B": [1000000000, 1, 1000000000, 433494437, 805306368, 999999937]}}
    for seed, n in enumerate([1, 256], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    yield {"name": "n-2000", "scalars": {"N": 2000}, "seed": 3, "processors": ["riscv-rvv", "coral-npu"]}
    yield {"name": "n-50000", "scalars": {"N": 50_000}, "seed": 4, "processors": ["riscv-rvv"]}


def generate(test, gen, device):
    n = test["N"]
    # pairs with a large common divisor, not only the coprime ones random numbers almost always are
    g = torch.randint(1, 30_001, (n,), dtype=torch.int64, device=device, generator=gen)
    a = torch.randint(1, 30_001, (n,), dtype=torch.int64, device=device, generator=gen)
    b = torch.randint(1, 30_001, (n,), dtype=torch.int64, device=device, generator=gen)
    return {"A": (g * a).to(torch.uint32), "B": (g * b).to(torch.uint32)}


def reference(test, inputs):
    return {"C": torch.gcd(inputs["A"].to(torch.int64), inputs["B"].to(torch.int64)).to(torch.uint32)}


def validate_input(test, inputs):
    return test["N"] >= 1 and all(
        bool((value.to(torch.int64) >= 1).all() and (value.to(torch.int64) <= 1_000_000_000).all())
        for value in inputs.values()
    )
