import torch


def tests():
    yield {"name": "sample", "sample": True, "scalars": {"M": 2, "K": 3, "N": 2},
           "inputs": {"A": [1, 2, 3, 4, 5, 6], "B": [1, 0, 0, 1, 1, 1]},
           "expected": {"C": [4, 5, 10, 11]}}
    # emulators count every cycle or instruction: up to 64x64 for rtl, 128x128 for riscv-rvv, the rest on the GPU.
    # The Coral NPU core has 32 KB of data memory for everything: three 48x48 matrices are 27 KB.
    sizes = [(1, 1, 1, None), (32, 32, 32, None), (256, 256, 256, ["gpu-cuda"]), (1000, 333, 777, ["gpu-cuda"]),
             (1000, 1000, 1000, ["gpu-cuda"]), (64, 64, 64, ["rtl", "riscv-rvv"]),
             (17, 40, 9, ["rtl", "riscv-rvv", "coral-npu"]), (128, 128, 128, ["riscv-rvv"]), (48, 48, 48, ["coral-npu"])]
    for seed, (m, k, n, processors) in enumerate(sizes, start=1):
        test = {"name": f"{m}x{k}x{n}", "scalars": {"M": m, "K": k, "N": n}, "seed": seed}
        if processors:
            test["processors"] = processors
        yield test


def generate(test, gen, device):
    m, k, n = test["M"], test["K"], test["N"]
    return {
        "A": torch.rand(m * k, generator=gen, device=device),
        "B": torch.rand(k * n, generator=gen, device=device),
    }


def reference(test, inputs):
    m, k, n = test["M"], test["K"], test["N"]
    # fp64 accumulation for the reference, stored as fp32
    a = inputs["A"].view(m, k).double()
    b = inputs["B"].view(k, n).double()
    return {"C": (a @ b).float().reshape(-1)}
