import torch

HIGH = 1_000_000
MAX_N = 1024
# the reference adds a block of rows of A to the whole B at a time: 64 MB of sums, never an N x N x N tensor
BLOCK_ELEMENTS = 16_000_000


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 3},
           "inputs": {"A": [4, 9, 2,
                            7, 1, 8,
                            5, 5, 5],
                      "B": [3, 6, 1,
                            2, 2, 9,
                            8, 1, 4]},
           "expected": {"C": [7, 3, 5,
                              3, 3, 8,
                              7, 6, 6]}}
    # the only way costs two million: more than any single ticket
    yield {"name": "n-1", "scalars": {"N": 1}, "inputs": {"A": [HIGH], "B": [HIGH]}, "expected": {"C": [2 * HIGH]},
           "measure": False}
    yield {"name": "all-high", "scalars": {"N": 3}, "inputs": {"A": [HIGH] * 9, "B": [HIGH] * 9}, "measure": False}
    yield {"name": "free", "scalars": {"N": 3}, "inputs": {"A": [0] * 9, "B": [0] * 9}, "measure": False}
    # the best change is the first station for some pairs and the last one for others
    yield {"name": "first-and-last", "scalars": {"N": 4},
           "inputs": {"A": [0, 5, 5, 5,
                            5, 5, 5, 0,
                            9, 9, 9, 1,
                            1, 9, 9, 9],
                      "B": [1, 2, 3, 4,
                            50, 60, 70, 80,
                            90, 10, 11, 12,
                            4, 3, 2, 1]},
           "expected": {"C": [1, 2, 3, 4,
                              4, 3, 2, 1,
                              5, 4, 3, 2,
                              2, 3, 4, 5]}, "measure": False}
    # B is not symmetric: the train goes from station k to city j, not back
    yield {"name": "one-way", "scalars": {"N": 2},
           "inputs": {"A": [1, 100, 100, 1], "B": [10, 2000, 30000, 400000]},
           "expected": {"C": [11, 2001, 110, 2100]}, "measure": False}
    yield {"name": "n-7", "scalars": {"N": 7}, "seed": 2, "measure": False}
    yield {"name": "n-33", "scalars": {"N": 33}, "seed": 3, "measure": False}
    # every ticket is close to a million: every answer is above a million
    yield {"name": "expensive-21", "scalars": {"N": 21}, "seed": 4, "measure": False}
    yield {"name": "n-32", "scalars": {"N": 32}, "seed": 5}
    # the Coral NPU core has 32 KB of data memory for everything: three matrices of 48 x 48 are 27 KB
    yield {"name": "n-48", "scalars": {"N": 48}, "seed": 6, "processors": ["coral-npu"]}
    # the work grows as the cube of N: an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "n-64", "scalars": {"N": 64}, "seed": 7, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "n-128", "scalars": {"N": 128}, "seed": 8, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "n-500", "scalars": {"N": 500}, "seed": 10, "processors": ["gpu-cuda"]}
    yield {"name": "expensive-777", "scalars": {"N": 777}, "seed": 11, "processors": ["gpu-cuda"]}
    yield {"name": "n-1024", "scalars": {"N": 1024}, "seed": 12, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    n = test["N"]
    low = 900_000 if test.name.startswith("expensive") else 0
    return {name: torch.randint(low, HIGH + 1, (n * n,), dtype=torch.int32, device=device, generator=gen)
            for name in ("A", "B")}


def reference(test, inputs):
    n = test["N"]
    a = inputs["A"].reshape(n, n)
    b = inputs["B"].reshape(n, n)
    c = torch.empty_like(a)
    rows = max(1, BLOCK_ELEMENTS // (n * n))
    for start in range(0, n, rows):
        c[start:start + rows] = (a[start:start + rows, :, None] + b[None, :, :]).amin(dim=1)
    return {"C": c.reshape(-1)}


def validate_input(test, inputs):
    return 1 <= test["N"] <= MAX_N and all(
        bool((value >= 0).all() and (value <= HIGH).all()) for value in inputs.values())
