import torch

MAX_N = 10_000_000
TOP = 2**32 - 1


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 11},
           "inputs": {"A": [0, 1, 8, 9, 15, 16, 999999, 1000000, 4294836224, 4294836225, 4294967295]},
           "expected": {"C": [0, 1, 2, 3, 3, 4, 999, 1000, 65534, 65535, 65535]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [TOP]}}
    yield {"name": "small", "scalars": {"N": 40}, "inputs": {"A": list(range(40))}, "measure": False}
    # k * k - 1, k * k and k * k + 1 where a 24-bit mantissa or a 32-bit (k + 1) * (k + 1) gives way
    roots = [2, 3, 255, 256, 1000, 4095, 4096, 4097, 5793, 11585, 23170, 32767, 32768, 46340, 46341, 65534, 65535]
    around = [k * k + d for k in roots for d in (-1, 0, 1)]
    yield {"name": "around-squares", "scalars": {"N": len(around)}, "inputs": {"A": around}, "measure": False}
    # the top of the range: 65535 * 65535 = 4294836225, and 65536 * 65536 no longer fits 32 bits
    top = [TOP, TOP - 1, 4294836226, 4294836225, 4294836224, 4294705156, 4294705155, 2147483648, 2147483647,
           2147395600, 2147395599, 16777216, 16777215, 16785409, 16785408]
    yield {"name": "top", "scalars": {"N": len(top)}, "inputs": {"A": top}, "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: a couple of thousand elements for rtl
    yield {"name": "n-2048", "scalars": {"N": 2048}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, MAX_N], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # A third of the numbers is uniform over 32 bits, a third has a uniform bit length, and a third sits right at a
    # perfect square: k * k - 1, k * k or k * k + 1.
    n = test["N"]
    uniform = torch.randint(0, 2**32, (n,), dtype=torch.int64, device=device, generator=gen)
    shift = torch.randint(0, 32, (n,), dtype=torch.int64, device=device, generator=gen)
    k = torch.randint(1, 65536, (n,), dtype=torch.int64, device=device, generator=gen)
    delta = torch.randint(-1, 2, (n,), dtype=torch.int64, device=device, generator=gen)
    kind = torch.randint(0, 3, (n,), dtype=torch.int64, device=device, generator=gen)
    a = torch.where(kind == 0, uniform, torch.where(kind == 1, uniform >> shift, k * k + delta))
    return {"A": a.to(torch.uint32)}


def reference(test, inputs):
    # float64 holds every 32-bit number exactly, and the two corrections make the answer independent of rounding
    a = inputs["A"].to(torch.int64) & 0xFFFFFFFF
    r = torch.sqrt(a.to(torch.float64)).to(torch.int64)
    r -= (r * r > a).to(torch.int64)
    r += ((r + 1) * (r + 1) <= a).to(torch.int64)
    return {"C": r.to(torch.uint32)}


def validate_input(test, inputs):
    return 1 <= test["N"] <= MAX_N
