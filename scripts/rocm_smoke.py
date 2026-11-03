"""ROCm smoke test: device visible, float32/float64 matmul + exp correctness and rough throughput."""
import time

import torch

print("torch", torch.__version__, "hip", torch.version.hip, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    d = torch.device("cuda")
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
    for dt in (torch.float32, torch.float64):
        a = torch.randn(2048, 2048, dtype=dt, device=d)
        b = torch.randn(2048, 2048, dtype=dt, device=d)
        c = a @ b; torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize(); dtm = (time.time() - t) / 10
        ref = (a.cpu().double() @ b.cpu().double())
        err = ((c.cpu().double() - ref).abs().max() / ref.abs().max()).item()
        x = torch.linspace(-50, 50, 10**6, dtype=dt, device=d)
        e = (torch.exp(x).cpu().double() / torch.exp(x.cpu().double()) - 1).abs().max().item()
        print(f"{dt}: matmul {2*2048**3/dtm/1e9:.1f} GFLOP/s rel err {err:.2e}; exp max rel err {e:.2e}")
