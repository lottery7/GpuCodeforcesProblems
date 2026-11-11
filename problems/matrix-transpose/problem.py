import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"N": 2, "M": 3},
           "inputs": {"A": [1, 2, 3, 4, 5, 6]},
           "expected": {"B": [1, 4, 2, 5, 3, 6]}}
    # the Coral NPU core has 32 KB of data memory for everything: two matrices of 3360 floats are 27 KB.
    # 1024x1024 does not fit the 1 MiB L2 of the riscv-rvv cache model: the order of the walk decides the score there.
    sizes = [(1, 1, None), (1, 37, None), (41, 1, None), (60, 56, ["coral-npu"]), (17, 190, ["coral-npu"]),
             (256, 200, ["cuda-sim"]), (1024, 1024, ["riscv-rvv"]), (700, 1000, ["riscv-rvv"]),
             (4096, 4096, ["gpu-cuda"]), (3000, 5000, ["gpu-cuda"])]
    for seed, (n, m, processors) in enumerate(sizes, start=1):
        test = {"name": f"{n}x{m}", "scalars": {"N": n, "M": m}, "seed": seed}
        if processors:
            test["processors"] = processors
        yield test


def generate(test, gen, device):
    return {"A": torch.rand(test["N"] * test["M"], generator=gen, device=device)}


def reference(test, inputs):
    return {"B": inputs["A"].view(test["N"], test["M"]).t().contiguous().reshape(-1)}


def validate_input(test, inputs):
    a = inputs["A"]
    return test["N"] >= 1 and test["M"] >= 1 and bool(torch.isfinite(a).all() and (a >= 0).all() and (a < 1).all())
