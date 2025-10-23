import numpy as np, torch, rasterio
from rasterio.windows import Window
from pathlib import Path
from .model import SiameseUNet
import matplotlib.pyplot as plt

def load_model(ckpt_path, in_ch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseUNet(in_ch=in_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, device

def infer_full_scene_rgb(t0_path, t1_path, ckpt_path, out_tif, in_ch=3, patch=512, thr=0.3):
    model, device = load_model(ckpt_path, in_ch)
    with rasterio.open(t0_path) as t0, rasterio.open(t1_path) as t1:
        H, W = t0.height, t0.width
        profile = t0.profile.copy(); profile.update(count=1, dtype="uint8")
        out = np.zeros((H, W), dtype=np.uint8)
        for y in range(0, H, patch):
            for x in range(0, W, patch):
                w = Window(x, y, min(patch, W-x), min(patch, H-y))
                a_np = t0.read([1,2,3], window=w).astype(np.float32)/255.0
                b_np = t1.read([1,2,3], window=w).astype(np.float32)/255.0
                a = torch.from_numpy(a_np).unsqueeze(0).to(device)
                b = torch.from_numpy(b_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model(a,b)).squeeze().cpu().numpy()
                out[y:y+w.height, x:x+w.width] = (p>thr).astype(np.uint8)[:w.height,:w.width]
        Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst: dst.write(out, 1)
    return out

def save_preview_png(t0_path, mask, out_png, max_side=1200):
    with rasterio.open(t0_path) as src:
        h,w = src.height, src.width
        scale = max(1, int(max(h,w)/max_side))
        rgb = src.read([1,2,3],
                       out_shape=(3, h//scale, w//scale))
    rgb = np.transpose(rgb,(1,2,0))
    rgb = (rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6)
    mask_ds = mask[::scale, ::scale]
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.imshow(rgb); plt.axis("off"); plt.title("Time 0 (downsampled)")
    plt.subplot(1,2,2); plt.imshow(mask_ds, cmap="Reds"); plt.axis("off"); plt.title("Predicted change")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()