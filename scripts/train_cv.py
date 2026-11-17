"""
Cross-validated training on the cached 484-case Decathlon cohort.
Reads fold assignments from folds_seed42_3fold.json; trains one fold per
invocation. Automatically resumes from the saved checkpoint if one exists,
so an interrupted run costs at most one epoch.

Usage: python scripts/train_cv.py --fold 0 --epochs 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(8)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ai_models.segmentation import UNet3D
from monai.losses import DiceCELoss

CACHE = PROJECT_ROOT / "data/brats_dataset/cache_128"
FOLDS = PROJECT_ROOT / "data/brats_dataset/folds_seed42_3fold.json"
CKPT_DIR = PROJECT_ROOT / "models/segmentation"
RESULTS_DIR = PROJECT_ROOT / "results/metrics"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = (16, 32, 64, 128, 256)
PATCH = 64
PATCHES_PER_CASE = 2
FG_RATIO = 1 / 3            # 1:2 foreground:background sampling
THRESHOLD = 0.5             # fixed globally a priori


def remap(lbl):
    """Decathlon 1=edema, 2=non-enhancing, 3=enhancing -> 0=bg, 1=tumour, 2=edema"""
    out = np.zeros_like(lbl)
    out[(lbl == 2) | (lbl == 3)] = 1
    out[lbl == 1] = 2
    return out


class PatchDS(Dataset):
    def __init__(self, ids, rng_seed=0):
        self.ids = ids
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.ids) * PATCHES_PER_CASE

    def __getitem__(self, i):
        cid = self.ids[i // PATCHES_PER_CASE]
        img = np.load(CACHE / "images" / f"{cid}.npy").astype(np.float32)
        lbl = remap(np.load(CACHE / "labels" / f"{cid}.npy"))

        want_fg = self.rng.random() < FG_RATIO
        fg = np.argwhere(lbl > 0)
        if want_fg and len(fg):
            c = fg[self.rng.integers(len(fg))]
            s = [int(np.clip(c[d] - PATCH // 2, 0, 128 - PATCH)) for d in range(3)]
        else:
            s = [int(self.rng.integers(0, 128 - PATCH + 1)) for _ in range(3)]

        sl = tuple(slice(s[d], s[d] + PATCH) for d in range(3))
        return (torch.from_numpy(img[(slice(None),) + sl].copy()),
                torch.from_numpy(lbl[sl].copy()).long())


@torch.no_grad()
def evaluate(model, ids):
    """Full-volume tumour-class Dice at the fixed global threshold."""
    model.eval()
    dices = []
    for cid in ids:
        img = torch.from_numpy(
            np.load(CACHE / "images" / f"{cid}.npy").astype(np.float32)).unsqueeze(0)
        lbl = remap(np.load(CACHE / "labels" / f"{cid}.npy"))
        prob = torch.softmax(model(img), dim=1)[0, 1].numpy()
        pred = prob > THRESHOLD
        gt = lbl == 1
        denom = pred.sum() + gt.sum()
        dices.append(2 * (pred & gt).sum() / denom if denom > 0 else np.nan)
    model.train()
    return np.array(dices, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    cfg = json.loads(FOLDS.read_text())
    fold = cfg["folds"][args.fold]
    tr, va = fold["train"], fold["val"]
    print(f"Fold {args.fold}: {len(tr)} train / {len(va)} val", flush=True)

    model = UNet3D(features=FEATURES)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ---- resume from checkpoint if present ----
    ckpt_path = CKPT_DIR / f"unet3d_cv_fold{args.fold}.pth"
    start_ep = 1
    best = -1.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
        best = ck.get("val_dice", -1.0)
        start_ep = ck.get("epoch", 0) + 1
        print(f"  resumed from epoch {ck.get('epoch')} "
              f"(best dice {best:.4f}); continuing at epoch {start_ep}", flush=True)

    if start_ep > args.epochs:
        print(f"  fold {args.fold} already complete "
              f"({start_ep-1} >= {args.epochs} epochs). Nothing to do.", flush=True)
        return

    dl = DataLoader(PatchDS(tr, rng_seed=args.fold), batch_size=1,
                    shuffle=True, num_workers=2)

    t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        tot, n = 0.0, 0
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y.unsqueeze(1))
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1

        d = evaluate(model, va)
        mean_d = float(np.nanmean(d))
        print(f"  ep {ep:2d}/{args.epochs}  loss={tot/n:.4f}  "
              f"val_dice={mean_d:.4f}  [{(time.time()-t0)/60:.0f} min this run]",
              flush=True)

        if mean_d > best:
            best = mean_d
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "fold": args.fold, "epoch": ep, "val_dice": mean_d,
                        "features": FEATURES, "threshold": THRESHOLD},
                       ckpt_path)

    # ---- final evaluation using the BEST checkpoint, not the last epoch ----
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    d = evaluate(model, va)

    out = RESULTS_DIR / f"cv_fold{args.fold}.json"
    out.write_text(json.dumps({
        "fold": args.fold,
        "epochs_requested": args.epochs,
        "best_epoch": ck.get("epoch"),
        "best_val_dice": best,
        "n_train": len(tr),
        "n_val": len(va),
        "threshold": THRESHOLD,
        "features": list(FEATURES),
        "per_case_dice": dict(zip(va, d.tolist())),
        "mean": float(np.nanmean(d)),
        "std": float(np.nanstd(d)),
        "median": float(np.nanmedian(d)),
        "n_nan": int(np.isnan(d).sum()),
    }, indent=2))

    print(f"Fold {args.fold} done. best={best:.4f} at epoch {ck.get('epoch')}  "
          f"({(time.time()-t0)/3600:.1f}h this run)", flush=True)


if __name__ == "__main__":
    main()
