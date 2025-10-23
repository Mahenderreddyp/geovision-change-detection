import os
from pathlib import Path
from geovision.infer import infer_full_scene_rgb, save_preview_png

OUT_PATH  = os.environ.get("OUT_PATH", "outputs")
INTERIM   = os.environ.get("INTERIM_PATH", "interim")
RUNS_PATH = os.environ.get("RUNS_PATH", "runs")
CKPT      = f"{RUNS_PATH}/best_model.pt"

AOIS = ["Chicago", "Houston", "SanJose"]

if __name__ == "__main__":
    for name in AOIS:
        t0 = f"{INTERIM}/{name}/t0_aligned.tif"
        t1 = f"{INTERIM}/{name}/t1_aligned.tif"
        out_tif = f"{OUT_PATH}/{name}/change_mask.tif"
        out_png = f"{OUT_PATH}/{name}/preview.png"
        Path(f"{OUT_PATH}/{name}").mkdir(parents=True, exist_ok=True)
        mask = infer_full_scene_rgb(t0, t1, CKPT, out_tif, in_ch=3, patch=512, thr=0.3)
        save_preview_png(t0, mask, out_png)
        print(f"✅ {name}: saved {out_tif} and {out_png}")