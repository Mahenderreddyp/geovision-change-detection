import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .data import PatchDataset
from .model import SiameseUNet

def iou_metric(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p*y).sum((1,2,3)); union = (p + y - p*y).sum((1,2,3))
    return ((inter+1e-6)/(union+1e-6)).mean().item()

def train_model(x_path, y_path, ckpt_out, epochs=100, batch_size=8, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PatchDataset(x_path, y_path)
    n  = len(ds)
    tr, val = random_split(ds, [int(0.8*n), n-int(0.8*n)])
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val_dl= DataLoader(val, batch_size=batch_size)

    in_ch = ds[0][0].shape[0]  # 3
    model = SiameseUNet(in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    for e in range(1, epochs+1):
        model.train(); tl=0
        for t0,t1,y in tr_dl:
            t0,t1,y = t0.to(device),t1.to(device),y.to(device)
            out = model(t0,t1); loss = loss_fn(out,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        model.eval(); vl=0; vi=0
        with torch.no_grad():
            for t0,t1,y in val_dl:
                t0,t1,y = t0.to(device),t1.to(device),y.to(device)
                out = model(t0,t1)
                vl += loss_fn(out,y).item()
                vi += iou_metric(out,y)
        vl/=len(val_dl); vi/=len(val_dl); tl/=len(tr_dl)
        print(f"Epoch {e:03d}: train_loss={tl:.3f} val_loss={vl:.3f} val_iou={vi:.3f}")
        if vi>best:
            best=vi
            torch.save(model.state_dict(), ckpt_out)
            print(f"✅ Saved new best model → {ckpt_out} (IoU={best:.3f})")