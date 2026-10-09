#!/usr/bin/env python3
"""Can each baseline train natively at 1 km on the regional window (768 x 1536)? One bf16 forward + backward, batch 1
and 2, peak GPU memory and time. Same architectures as train_baselines.py, only img_size changes."""
import sys, time
import torch
sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain")
import train_baselines as TB

H, W, C = 768, 1536, 71 + 31
dev = torch.device("cuda")
for name in sys.argv[1:]:
    TB.CH, TB.CW = H, W
    for B in (1, 2):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            model = {
                "climax": lambda c: TB.ClimaXBaseline(in_channels=c, img_size=(H, W), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1),
                "earthformer": lambda c: TB.EarthformerBaseline(in_channels=c, img_size=(H, W), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1, window_size=(3, 5)),
                "convlstm": lambda c: TB.ConvLSTMBaseline(in_channels=c, img_size=(H, W), hidden_channels=(128, 256), kernel_size=3, drop_rate=0.1),
                "simvp": lambda c: TB.SimVPBaseline(in_channels=c, img_size=(H, W), hidden_channels=256, translator_layers=8, reduction=4, drop_rate=0.1),
                "topoflow": lambda c: TB.TopoFlowBaseline(in_channels=c, img_size=(H, W), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1, out_channels=1),
            }[name](C).to(dev)
            x = torch.randn(B, C - 1, H, W, device=dev); g = torch.randn(B, 1, H, W, device=dev)  # in_channels counts GHAP
            e = torch.randn(B, H, W, device=dev); l = torch.ones(B, device=dev)
            t = time.time()
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, g, e, l)
                out.float().mean().backward()
            torch.cuda.synchronize()
            print(f"{name:12s} B={B}: OK  peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GB  {(time.time()-t)/2:.2f} s/step  "
                  f"out {tuple(out.shape)}  params {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{name:12s} B={B}: OUT OF MEMORY", flush=True)
        except Exception as ex:
            print(f"{name:12s} B={B}: ERROR {type(ex).__name__}: {str(ex)[:200]}", flush=True)
        model = None
