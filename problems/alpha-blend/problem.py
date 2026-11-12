import torch

MAX_SIZE = 10_000_000


def corners():
    # every mask value against every pair of the pixels 0, 1, 128 and 255: 4 * 4 * 256 = 4096 triples
    pixels = torch.tensor([0, 1, 128, 255], dtype=torch.uint8)
    a, b, t = torch.meshgrid(pixels, pixels, torch.arange(256, dtype=torch.uint8), indexing="ij")
    return {"A": a.reshape(-1), "B": b.reshape(-1), "T": t.reshape(-1)}


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"A": [200, 200, 0, 255, 100, 100], "B": [50, 50, 255, 0, 101, 101], "T": [0, 255, 128, 128, 127, 128]},
           "expected": {"C": [200, 50, 128, 127, 100, 101]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [10], "B": [20], "T": [51]}, "measure": False}
    # a black mask shows the first shot untouched, a white one the second
    yield {"name": "t-0", "scalars": {"N": 5},
           "inputs": {"A": [0, 1, 254, 255, 77], "B": [255, 9, 0, 0, 200], "T": [0] * 5}, "measure": False}
    yield {"name": "t-255", "scalars": {"N": 5},
           "inputs": {"A": [0, 1, 254, 255, 77], "B": [255, 9, 0, 0, 200], "T": [255] * 5}, "measure": False}
    # two equal shots stay what they are under any mask
    yield {"name": "same", "scalars": {"N": 6},
           "inputs": {"A": [255, 255, 1, 1, 128, 0], "B": [255, 255, 1, 1, 128, 0], "T": [1, 254, 127, 128, 200, 77]},
           "measure": False}
    yield {"name": "corners", "scalars": {"N": 4096}, "inputs": corners()}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: a 64x64 picture for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: four planes of 6000 bytes fit
    yield {"name": "n-6000", "scalars": {"N": 6000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # A mask is mostly black or white with soft edges: a quarter of it is 0, a quarter 255, the rest anything.
    n = test["N"]
    a, b = (torch.randint(0, 256, (n,), dtype=torch.uint8, device=device, generator=gen) for _ in range(2))
    t = torch.randint(-128, 384, (n,), device=device, generator=gen).clamp(0, 255).to(torch.uint8)
    return {"A": a, "B": b, "T": t}


def reference(test, inputs):
    a, b, t = (inputs[name].to(torch.int32) for name in ("A", "B", "T"))
    mixed = a * (255 - t) + b * t + 127
    return {"C": torch.div(mixed, 255, rounding_mode="floor").to(torch.uint8)}


def validate_input(test, inputs):
    return 1 <= test["N"] <= MAX_SIZE
