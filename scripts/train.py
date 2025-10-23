import os
from pathlib import Path
from geovision.train import train_model

PROC_PATH = os.environ.get("PROC_PATH", "processed")
RUNS_PATH = os.environ.get("RUNS_PATH", "runs")
Path(RUNS_PATH).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    x = f"{PROC_PATH}/X.npy"
    y = f"{PROC_PATH}/y.npy"
    ckpt = f"{RUNS_PATH}/best_model.pt"
    train_model(x, y, ckpt_out=ckpt, epochs=120, batch_size=8, lr=1e-3)