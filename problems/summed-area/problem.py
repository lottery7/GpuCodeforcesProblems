import torch

LOW, HIGH = -100, 100
# 16 777 216 cells of at most 100 in absolute value: every sum stays within 1 677 721 600, inside INT32
MAX_CELLS = 16_777_216


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 3, "M": 4},
           "inputs": {"A": [3, -1, 4, 0,
                            -2, 5, -6, 2,
                            1, 0, 7, -3]},
           "expected": {"S": [3, 2, 6, 6,
                              1, 5, 3, 5,
                              2, 6, 11, 10]}}
    yield {"name": "1x1", "scalars": {"N": 1, "M": 1}, "inputs": {"A": [LOW]}, "measure": False}
    # one row and one column: a prefix sum in one direction only
    yield {"name": "row", "scalars": {"N": 1, "M": 7}, "inputs": {"A": [5, -9, 100, -100, 0, 42, -1]},
           "measure": False}
    yield {"name": "column", "scalars": {"N": 6, "M": 1}, "inputs": {"A": [-7, 7, 100, 100, -50, 3]},
           "measure": False}
    # every cell at a bound: the sums grow as fast as they can, with either sign
    yield {"name": "all-high", "scalars": {"N": 40, "M": 25}, "inputs": {"A": [HIGH] * 1000}, "measure": False}
    yield {"name": "all-low", "scalars": {"N": 25, "M": 40}, "inputs": {"A": [LOW] * 1000}, "measure": False}
    # not square, and neither side is a multiple of a vector length
    yield {"name": "37x19", "scalars": {"N": 37, "M": 19}, "seed": 2, "measure": False}
    yield {"name": "5x203", "scalars": {"N": 5, "M": 203}, "seed": 3, "measure": False}
    yield {"name": "32x48", "scalars": {"N": 32, "M": 48}, "seed": 4}
    # the Coral NPU core has 32 KB of data memory for everything: two tables of 3360 numbers are 27 KB
    yield {"name": "56x60", "scalars": {"N": 56, "M": 60}, "seed": 5, "processors": ["coral-npu"]}
    yield {"name": "16x210", "scalars": {"N": 16, "M": 210}, "seed": 6, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "250x400", "scalars": {"N": 250, "M": 400}, "seed": 7, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "500x200", "scalars": {"N": 500, "M": 200}, "seed": 8, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "1000x1000", "scalars": {"N": 1000, "M": 1000}, "seed": 10, "processors": ["gpu-cuda"]}
    yield {"name": "4096x4096", "scalars": {"N": 4096, "M": 4096}, "seed": 11, "processors": ["gpu-cuda"]}
    yield {"name": "3000x5000", "scalars": {"N": 3000, "M": 5000}, "seed": 12, "processors": ["gpu-cuda"]}
    yield {"name": "8x2000000", "scalars": {"N": 8, "M": 2_000_000}, "seed": 13, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    return {"A": torch.randint(LOW, HIGH + 1, (test["N"] * test["M"],), dtype=torch.int32, device=device,
                               generator=gen)}


def reference(test, inputs):
    table = inputs["A"].reshape(test["N"], test["M"])
    return {"S": table.cumsum(dim=0, dtype=torch.int32).cumsum(dim=1, dtype=torch.int32).reshape(-1)}


def validate_input(test, inputs):
    a = inputs["A"]
    n, m = test["N"], test["M"]
    return n >= 1 and m >= 1 and n * m <= MAX_CELLS and bool((a >= LOW).all() and (a <= HIGH).all())
