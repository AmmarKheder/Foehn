import sys, time, torch
sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain")
import train_baselines as TB
dev = torch.device("cuda")
for (H, W, C, B) in ((168, 280, 102, 4), (768, 1536, 74, 2), (768, 1536, 74, 1)):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = TB.ClimaXBaselineV2(in_channels=C, img_size=(H, W), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1).to(dev).train()
        x, g, e, l = torch.randn(B, C - 1, H, W, device=dev), torch.randn(B, 1, H, W, device=dev), torch.randn(B, H, W, device=dev), torch.ones(B, device=dev)
        for k in range(4):
            if k == 1: torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(x, g, e, l)
            out.float().mean().backward()
        torch.cuda.synchronize()
        print(f"{H}x{W} C={C} B={B}: OK peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB  {(time.time()-t)/3:.2f} s/step  out {tuple(out.shape)}  params {sum(p.numel() for p in m.parameters())/1e6:.0f}M", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{H}x{W} C={C} B={B}: OUT OF MEMORY", flush=True)
    m = None
