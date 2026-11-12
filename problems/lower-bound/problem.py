import torch

LOW, HIGH = -1_000_000_000, 1_000_000_000
MAX_SIZE = 10_000_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 8, "M": 7},
           "inputs": {"A": [-20, -5, -5, 0, 10, 10, 10, 35], "Q": [10, -30, 36, 0, -5, 11, 35]},
           "expected": {"C": [4, 0, 8, 3, 1, 7, 7]}}
    yield {"name": "n-1", "scalars": {"N": 1, "M": 5}, "inputs": {"A": [7], "Q": [6, 7, 8, LOW, HIGH]}, "measure": False}
    # one question above every total: the answer is N itself
    yield {"name": "m-1", "scalars": {"N": 5, "M": 1}, "inputs": {"A": [-3, -3, -1, 4, 4], "Q": [5]}, "measure": False}
    yield {"name": "n-1-m-1", "scalars": {"N": 1, "M": 1}, "inputs": {"A": [0], "Q": [1]}, "measure": False}
    yield {"name": "all-equal", "scalars": {"N": 6, "M": 3}, "inputs": {"A": [4] * 6, "Q": [3, 4, 5]}, "measure": False}
    # the bounds of the values: a comparison must be signed and must not go through a difference
    yield {"name": "extremes", "scalars": {"N": 5, "M": 6},
           "inputs": {"A": [LOW, LOW, 0, HIGH, HIGH], "Q": [LOW, HIGH, LOW + 1, HIGH - 1, 0, 1]}, "measure": False}
    yield {"name": "n-1000", "scalars": {"N": 1000, "M": 1000}, "seed": 2}
    # a circuit is simulated cycle by cycle, and every question costs it about log N reads
    yield {"name": "n-4096-m-1024", "scalars": {"N": 4096, "M": 1024}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: 12000 bytes of A and twice 6000 of Q and C fit
    yield {"name": "n-3000-m-1500", "scalars": {"N": 3000, "M": 1500}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000, "M": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, size in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{size}", "scalars": {"N": size, "M": size}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # Totals sit on a grid with about N / 2 points, so ties are everywhere. Half of the questions name a total of
    # the table, a quarter miss one by a point, the rest are anywhere, below the first and above the last included.
    n, m = test["N"], test["M"]
    points = max(1, n // 2)
    step = 1_800_000_000 // points
    a = torch.randint(0, points, (n,), device=device, generator=gen) * step - 900_000_000
    a = a.sort().values
    picked = a[torch.randint(0, n, (m,), device=device, generator=gen)]
    kind = torch.randint(0, 4, (m,), device=device, generator=gen)
    nudge = torch.randint(0, 2, (m,), device=device, generator=gen) * 2 - 1
    anywhere = torch.randint(LOW, HIGH + 1, (m,), device=device, generator=gen)
    q = torch.where(kind < 2, picked, torch.where(kind == 2, picked + nudge, anywhere))
    return {"A": a.to(torch.int32), "Q": q.to(torch.int32)}


def reference(test, inputs):
    return {"C": torch.searchsorted(inputs["A"], inputs["Q"], right=False).to(torch.int32)}


def validate_input(test, inputs):
    a, q = inputs["A"], inputs["Q"]
    return (1 <= test["N"] <= MAX_SIZE and 1 <= test["M"] <= MAX_SIZE
            and bool((a >= LOW).all() and (a <= HIGH).all() and (q >= LOW).all() and (q <= HIGH).all())
            and bool((a[1:] >= a[:-1]).all()))
