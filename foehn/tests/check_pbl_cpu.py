"""mixing heights of flexgreen.derive on ERA5 (CPU): sites and global percentiles, January and July 2017"""
import sys
import numpy as np, torch
sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km")
import flexgreen as fg
dev = torch.device("cpu")
zs, z0 = fg.static(dev)
for doy, tag in ((0, "Jan"), (181, "Jul")):
    B = torch.tensor(fg.load_run("era5", 2017, doy)[:8], device=dev)
    f = fg.derive(B, zs, z0)
    h = f[:, fg.C_PBL].numpy()
    lsm = np.load(fg.WB / "static_1deg.npz")["lsm"]
    for nm, (la, lo) in {"Beijing": (40, 116), "Sahara": (17, 8), "Delhi": (29, 77), "Paris": (49, 2), "S.Pacific": (-20, -130),
                         "N.Atlantic": (40, -40), "Amazon": (-3, -60), "Siberia": (62, 100)}.items():
        r, c = int(90 - la), int(lo + 180)
        print(f"{tag} {nm:10s} h (00/06/12/18 UTC, 2 days): {np.round(h[:8, r, c]).astype(int)}")
    land = lsm > 0.5
    print(f"{tag} percentiles 10/50/90 %: land {np.percentile(h[:, land], [10, 50, 90]).round(0)} | ocean {np.percentile(h[:, ~land], [10, 50, 90]).round(0)}")
