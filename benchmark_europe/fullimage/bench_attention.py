"""Can one LUMI GCD run exact global self-attention over the whole 1 km Europe grid (4192 x 6992 = 29.3 M pixels)?
Patch p -> N = (4192/p) * (6992/p) tokens. Forward + backward of one attention layer (bf16, 12 heads x 64), SDPA."""
import time
import torch
import torch.nn.functional as F

dev = torch.device("cuda")
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
for p in (32, 23, 16):
    N = (4192 // p) * (6992 // p)
    for name, backend in (("default", None),):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            q = torch.randn(1, 12, N, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
            for it in range(3):
                torch.cuda.synchronize(); t0 = time.time()
                o = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize(); t1 = time.time()
                o.float().sum().backward()
                torch.cuda.synchronize(); t2 = time.time()
            gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"patch {p:2d} ({p} km): N = {N:7d} tokens | fwd {t1 - t0:6.2f} s  bwd {t2 - t1:6.2f} s | peak mem {gb:6.1f} GB", flush=True)
        except Exception as e:
            print(f"patch {p:2d}: N = {N:7d} -> FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        del q, k, v
