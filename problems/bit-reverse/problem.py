import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 4},
           "inputs": {"A": [1, 2, 2147483648, 240]},
           "expected": {"C": [2147483648, 1073741824, 1, 251658240]}}
    yield {"name": "edges", "scalars": {"N": 4}, "inputs": {"A": [0, 4294967295, 2863311530, 305419896]}, "measure": False}
    for seed, n in enumerate([1, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # torch draws no uint32 directly: draw the 32 bits in 64 and narrow them
    return {"A": torch.randint(0, 2**32, (test["N"],), dtype=torch.int64, device=device, generator=gen).to(torch.uint32)}


def reference(test, inputs):
    a = inputs["A"].to(torch.int64) & 0xFFFFFFFF
    c = torch.zeros_like(a)
    for bit in range(32):
        c |= ((a >> bit) & 1) << (31 - bit)
    return {"C": c.to(torch.uint32)}


def validate_input(test, inputs):
    return test["N"] >= 1
