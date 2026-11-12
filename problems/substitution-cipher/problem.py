import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 6},
           "inputs": {"T": [(7 * v + 3) % 256 for v in range(256)], "A": [72, 105, 33, 200, 0, 255]},
           "expected": {"C": [251, 226, 234, 123, 3, 252]}}
    yield {"name": "n-1", "scalars": {"N": 1},
           "inputs": {"T": [255 - v for v in range(256)], "A": [255]}, "measure": False}
    # the message is every byte value in order, so the answer is the table itself
    yield {"name": "alphabet", "scalars": {"N": 256},
           "inputs": {"T": [(167 * v + 13) % 256 for v in range(256)], "A": list(range(256))}, "measure": False}
    # bytes from 128 up: an index read as a signed char or cut to seven bits misses the table
    yield {"name": "high-bytes", "scalars": {"N": 9},
           "inputs": {"T": [v ^ 0x5A for v in range(256)], "A": [128, 255, 129, 254, 170, 213, 128, 200, 255]},
           "measure": False}
    # one byte value repeated: every lane of a gather reads the same address; the length is not a multiple of 16
    yield {"name": "one-letter", "scalars": {"N": 77},
           "inputs": {"T": [(v + 100) % 256 for v in range(256)], "A": [233] * 77}, "measure": False}
    yield {"name": "n-1000", "scalars": {"N": 1000}, "seed": 2}
    # a circuit is simulated cycle by cycle: thousands of bytes for rtl
    yield {"name": "n-4096", "scalars": {"N": 4096}, "seed": 3, "processors": ["rtl"]}
    # the Coral NPU core has 32 KB of data memory for everything: the table and two arrays of 12000 bytes fit
    yield {"name": "n-12000", "scalars": {"N": 12_000}, "seed": 4, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-100000", "scalars": {"N": 100_000}, "seed": 5, "processors": ["riscv-rvv", "cuda-sim"]}
    for seed, n in enumerate([1_000_000, 10_000_000], start=10):
        yield {"name": f"n-{n}", "scalars": {"N": n}, "seed": seed, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # The table of a test is a permutation, as a cipher needs; the message is uniform bytes.
    return {"T": torch.randperm(256, device=device, generator=gen).to(torch.uint8),
            "A": torch.randint(0, 256, (test["N"],), dtype=torch.uint8, device=device, generator=gen)}


def reference(test, inputs):
    return {"C": inputs["T"][inputs["A"].to(torch.int64)]}


def validate_input(test, inputs):
    # any 256 bytes are a table and any bytes are a message
    return test["N"] >= 1
