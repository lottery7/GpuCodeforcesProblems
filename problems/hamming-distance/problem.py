import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 3},
           "inputs": {"A": [0, 255, 5], "B": [1, 15, 5]},
           "expected": {"R": [5]}}
    for seed, n in enumerate([1, 1024], start=1):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=6):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


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


def validate_input(test, inputs):
    return test["N"] >= 1
