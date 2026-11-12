import torch
import torch.nn.functional as F

MAX_CELLS = 16_777_216


def tests():
    # a blinker cut by the top edge and three cells in the corner that grow into a block
    yield {"name": "sample", "sample": True, "scalars": {"N": 4, "M": 6},
           "inputs": {"A": [0, 0, 1, 1, 1, 0,
                            0, 0, 0, 0, 0, 0,
                            1, 0, 0, 0, 0, 0,
                            1, 1, 0, 0, 0, 0]},
           "expected": {"B": [0, 0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0, 0,
                              1, 1, 0, 0, 0, 0,
                              1, 1, 0, 0, 0, 0]}}
    yield {"name": "blinker", "scalars": {"N": 5, "M": 5},
           "inputs": {"A": [0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]},
           "expected": {"B": [0, 0, 0, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 0, 0, 0]}, "measure": False}
    yield {"name": "glider", "scalars": {"N": 6, "M": 6},
           "inputs": {"A": [0, 0, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0,
                            0, 0, 0, 1, 0, 0,
                            0, 1, 1, 1, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0]},
           "expected": {"B": [0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 1, 0, 1, 0, 0,
                              0, 0, 1, 1, 0, 0,
                              0, 0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0, 0]}, "measure": False}
    # a lonely cell dies, and with it the whole field: the answer is all zeros
    yield {"name": "1x1-alive", "scalars": {"N": 1, "M": 1}, "inputs": {"A": [1]}, "expected": {"B": [0]},
           "measure": False}
    yield {"name": "1x1-dead", "scalars": {"N": 1, "M": 1}, "inputs": {"A": [0]}, "expected": {"B": [0]},
           "measure": False}
    # a field one cell wide has no rows above or below (no columns to the sides): nothing is born there
    yield {"name": "row", "scalars": {"N": 1, "M": 9}, "inputs": {"A": [1, 1, 1, 0, 1, 1, 0, 1, 0]},
           "expected": {"B": [0, 1, 0, 0, 0, 0, 0, 0, 0]}, "measure": False}
    yield {"name": "column", "scalars": {"N": 7, "M": 1}, "inputs": {"A": [1, 1, 1, 1, 0, 1, 1]},
           "expected": {"B": [0, 1, 1, 0, 0, 0, 0]}, "measure": False}
    # every cell alive: only the corners have three neighbours and survive
    yield {"name": "all-alive", "scalars": {"N": 6, "M": 7}, "inputs": {"A": [1] * 42}, "measure": False}
    yield {"name": "block", "scalars": {"N": 2, "M": 2}, "inputs": {"A": [1, 1, 1, 1]},
           "expected": {"B": [1, 1, 1, 1]}, "measure": False}
    # live cells on the left and right edges: the end of a row is not a neighbour of the start of the next one
    yield {"name": "edges", "scalars": {"N": 3, "M": 4},
           "inputs": {"A": [1, 0, 0, 1,
                            1, 0, 0, 1,
                            1, 0, 0, 1]},
           "expected": {"B": [0, 0, 0, 0,
                              1, 1, 1, 1,
                              0, 0, 0, 0]}, "measure": False}
    # not square, and neither side is a multiple of a vector length
    yield {"name": "13x7", "scalars": {"N": 13, "M": 7}, "seed": 2, "measure": False}
    yield {"name": "3x101", "scalars": {"N": 3, "M": 101}, "seed": 3, "measure": False}
    yield {"name": "2x67", "scalars": {"N": 2, "M": 67}, "seed": 4, "measure": False}
    yield {"name": "32x48", "scalars": {"N": 32, "M": 48}, "seed": 5}
    # the Coral NPU core has 32 KB of data memory for everything: two fields of 12000 bytes fit
    yield {"name": "100x120", "scalars": {"N": 100, "M": 120}, "seed": 6, "processors": ["coral-npu"]}
    yield {"name": "24x500", "scalars": {"N": 24, "M": 500}, "seed": 7, "processors": ["coral-npu"]}
    # an emulated CPU counts every instruction, a simulated GPU every cycle
    yield {"name": "250x400", "scalars": {"N": 250, "M": 400}, "seed": 8, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "1000x100", "scalars": {"N": 1000, "M": 100}, "seed": 9, "processors": ["riscv-rvv", "cuda-sim"]}
    yield {"name": "1000x1000", "scalars": {"N": 1000, "M": 1000}, "seed": 10, "processors": ["gpu-cuda"]}
    yield {"name": "4096x4096", "scalars": {"N": 4096, "M": 4096}, "seed": 11, "processors": ["gpu-cuda"]}
    yield {"name": "3000x5000", "scalars": {"N": 3000, "M": 5000}, "seed": 12, "processors": ["gpu-cuda"]}


def generate(test, gen, device):
    # About a third of the cells alive: births, survivals and deaths are all frequent.
    cells = test["N"] * test["M"]
    return {"A": (torch.rand(cells, device=device, generator=gen) < 0.35).to(torch.uint8)}


def reference(test, inputs):
    n, m = test["N"], test["M"]
    field = inputs["A"].reshape(n, m)
    padded = F.pad(field, (1, 1, 1, 1))
    neighbours = torch.zeros_like(field)
    for dr in range(3):
        for dc in range(3):
            if (dr, dc) != (1, 1):
                neighbours += padded[dr:dr + n, dc:dc + m]
    alive = (neighbours == 3) | ((neighbours == 2) & (field == 1))
    return {"B": alive.to(torch.uint8).reshape(-1)}


def validate_input(test, inputs):
    n, m = test["N"], test["M"]
    return n >= 1 and m >= 1 and n * m <= MAX_CELLS and bool((inputs["A"] <= 1).all())
