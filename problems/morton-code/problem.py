import torch

MAX_N = 10_000_000
SIDE = 65536


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"X": [0, 1, 0, 3, 65535, 65535], "Y": [0, 0, 1, 5, 0, 65535]},
           "expected": {"C": [0, 1, 2, 39, 1431655765, 4294967295]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"X": [40000], "Y": [25000]}}
    # one bit of X or of Y at a time: every wire of the interleaving on its own
    yield {"name": "single-bits", "scalars": {"N": 32},
           "inputs": {"X": [1 << bit for bit in range(16)] + [0] * 16,
                      "Y": [0] * 16 + [1 << bit for bit in range(16)]},
           "measure": False}
    yield {"name": "corners", "scalars": {"N": 6},
           "inputs": {"X": [0, 65535, 0, 65535, 21845, 43690], "Y": [0, 0, 65535, 65535, 43690, 21845]},
           "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: three arrays of 8000 bytes fit
    yield {"name": "n-2000", "scalars": {"N": 2000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, MAX_N], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n = test["N"]
    return {name: torch.randint(0, SIDE, (n,), dtype=torch.int64, device=device, generator=gen).to(torch.uint32)
            for name in ("X", "Y")}


def reference(test, inputs):
    x = inputs["X"].to(torch.int64)
    y = inputs["Y"].to(torch.int64)
    c = torch.zeros_like(x)
    for bit in range(16):
        c |= ((x >> bit) & 1) << (2 * bit)
        c |= ((y >> bit) & 1) << (2 * bit + 1)
    return {"C": c.to(torch.uint32)}


def validate_input(test, inputs):
    x = inputs["X"].to(torch.int64) & 0xFFFFFFFF
    y = inputs["Y"].to(torch.int64) & 0xFFFFFFFF
    return 1 <= test["N"] <= MAX_N and bool((x < SIDE).all() and (y < SIDE).all())
