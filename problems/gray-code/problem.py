import torch

MAX_N = 10_000_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 10},
           "inputs": {"A": [0, 1, 3, 2, 6, 7, 5, 4, 12, 2147483648]},
           "expected": {"C": [0, 1, 2, 3, 4, 5, 6, 7, 8, 4294967295]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [4294967295]}}
    # one bit in every position: the answer is that bit and everything below it
    yield {"name": "single-bits", "scalars": {"N": 32}, "inputs": {"A": [1 << bit for bit in range(32)]},
           "measure": False}
    # all ones, alternating bits, the top bit alone and next to its neighbour, codes of 2^31 - 1 and 2^32 - 1
    yield {"name": "patterns", "scalars": {"N": 8},
           "inputs": {"A": [0, 4294967295, 2863311530, 1431655765, 3221225472, 1073741824, 2147483648, 305419896]},
           "measure": False}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # a circuit is simulated cycle by cycle: thousands of elements for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, MAX_N], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # torch draws no uint32 directly: draw the 32 bits in 64 and narrow them
    a = torch.randint(0, 2**32, (test["N"],), dtype=torch.int64, device=device, generator=gen)
    return {"A": a.to(torch.uint32)}


def reference(test, inputs):
    # bit k of the answer is the XOR of the bits k..31 of the code
    x = inputs["A"].to(torch.int64) & 0xFFFFFFFF
    for shift in (1, 2, 4, 8, 16):
        x ^= x >> shift
    return {"C": x.to(torch.uint32)}


def validate_input(test, inputs):
    return 1 <= test["N"] <= MAX_N
