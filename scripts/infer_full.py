import os
from pathlib import Path
from geovision.infer import infer_full_scene_rgb, save_preview_png

INTERIM_PATH = os.environ.get("INTERIM_PATH", "interim")
OUT_PATH     = os.environ.get("OUT_PATH", "outputs")
RUNS_PATH    = os.environ.get("RUNS_PATH", "runs")

if __name__ == "__main__":
    t0 = f"{INTERIM_PATH}/dataset_aligned/t0_aligned.tif"
    t1 = f"{INTERIM_PATH}/dataset_aligned/t1_aligned.tif"
    ckpt = f"{RUNS_PATH}/best_model.pt"
    out_tif = f"{OUT_PATH}/change_mask_inferred.tif"
    out_png = f"{OUT_PATH}/preview.png"
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    mask = infer_full_scene_rgb(t0, t1, ckpt, out_tif, in_ch=3, patch=512, thr=0.3)
    save_preview_png(t0, mask, out_png)
    print(f"✅ Saved: {out_tif} and {out_png}")