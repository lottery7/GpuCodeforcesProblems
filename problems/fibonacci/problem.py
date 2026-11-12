import torch

MAX_N = 10_000_000
MAX_A = 1_000_000_000
MASK = 0xFFFFFFFF


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 8},
           "inputs": {"A": [0, 1, 2, 3, 10, 47, 48, 1000000000]},
           "expected": {"C": [0, 1, 1, 2, 55, 2971215073, 512559680, 1532868155]}}
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [MAX_A]}}
    yield {"name": "small", "scalars": {"N": 64}, "inputs": {"A": list(range(64))}}
    # powers of two and their neighbours up to 2^29, the last power below the bound, and the bound itself
    powers = [(1 << bit) + d for bit in range(2, 30) for d in (-1, 0, 1)]
    yield {"name": "powers", "scalars": {"N": len(powers) + 4},
           "inputs": {"A": powers + [MAX_A - 1, MAX_A, 715827882, 357913941]}}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 2}
    # the Coral NPU core has 32 KB of data memory for everything: two arrays of 12000 bytes fit
    yield {"name": "n-3000", "scalars": {"N": 3000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, MAX_N], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # Mostly uniform up to the bound; every eighth number has a uniform bit length, so small ones are there too.
    n = test["N"]
    uniform = torch.randint(0, MAX_A + 1, (n,), dtype=torch.int64, device=device, generator=gen)
    shift = torch.randint(0, 30, (n,), dtype=torch.int64, device=device, generator=gen)
    short = torch.randint(0, 8, (n,), dtype=torch.int64, device=device, generator=gen) == 0
    return {"A": torch.where(short, uniform >> shift, uniform).to(torch.uint32)}


def reference(test, inputs):
    # Fast doubling over the 30 bits of the index, from the top one: (a, b) = (F(k), F(k + 1)) becomes
    # (F(2k), F(2k + 1)) and then steps by one where the bit is set. The operands are kept below 2^32; an int64
    # product of two such numbers may wrap, but its low 32 bits are still the low 32 bits of the true product.
    n = inputs["A"].to(torch.int64) & MASK
    a = torch.zeros_like(n)
    b = torch.ones_like(n)
    for bit in range(29, -1, -1):
        even = (a * ((2 * b - a) & MASK)) & MASK
        odd = (a * a + b * b) & MASK
        step = ((n >> bit) & 1).bool()
        a, b = torch.where(step, odd, even), torch.where(step, (even + odd) & MASK, odd)
    return {"C": a.to(torch.uint32)}


def validate_input(test, inputs):
    a = inputs["A"].to(torch.int64) & MASK
    return 1 <= test["N"] <= MAX_N and bool((a <= MAX_A).all())
