import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"A": [1, 2, 3, 6, 7, 27]},
           "expected": {"C": [0, 1, 7, 8, 16, 111]}}
    # the longest walk below a million (524 steps, up to 56 991 483 520 on the way), powers of two and the bounds
    yield {"name": "extremes", "scalars": {"N": 8},
           "inputs": {"A": [837799, 1000000, 999999, 524288, 1, 626331, 704511, 2]}}
    for seed, n in enumerate([1, 256], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit spends a cycle per step: a thousand numbers for rtl
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 3, "processors": ["rtl", "riscv-rvv", "cuda-sim", "coral-npu"]}
    yield {"name": "n-2000", "scalars": {"N": 2000}, "seed": 4, "processors": ["coral-npu"]}
    # every thread of a simulated GPU walks its own number of steps
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 5, "processors": ["cuda-sim"]}
    yield {"name": "n-50000", "scalars": {"N": 50_000}, "seed": 6, "processors": ["riscv-rvv"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {"A": torch.randint(1, 1_000_001, (test["N"],), dtype=torch.int64, device=device, generator=gen).to(torch.uint32)}


def reference(test, inputs):
    x = inputs["A"].to(torch.int64)
    steps = torch.zeros_like(x)
    while True:
        walking = x != 1
        if not bool(walking.any()):
            break
        x = torch.where(walking, torch.where(x % 2 == 0, x // 2, 3 * x + 1), x)
        steps += walking
    return {"C": steps.to(torch.int32)}


def validate_input(test, inputs):
    a = inputs["A"].to(torch.int64)
    return test["N"] >= 1 and bool((a >= 1).all() and (a <= 1_000_000).all())
