"""
Seeded, randomized 5-fold cross-validation splits over the full 484-case cohort.
Usage: python scripts/make_folds.py
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE = PROJECT_ROOT / "data/brats_dataset/cache_128"
OUT = PROJECT_ROOT / "data/brats_dataset/folds_seed42_3fold.json"

SEED = 42
N_FOLDS = 3


def main():
    case_ids = sorted(p.stem for p in (CACHE / "images").glob("*.npy"))
    print(f"Found {len(case_ids)} cached cases")

    rng = np.random.default_rng(SEED)
    shuffled = np.array(case_ids)
    rng.shuffle(shuffled)

    folds = []
    for k in range(N_FOLDS):
        val = shuffled[k::N_FOLDS].tolist()
        val_set = set(val)
        train = [c for c in shuffled.tolist() if c not in val_set]
        folds.append({"fold": k, "train": train, "val": val})
        print(f"  fold {k}: {len(train)} train / {len(val)} val")

    all_val = [c for f in folds for c in f["val"]]
    assert sorted(all_val) == sorted(case_ids), "validation coverage error"
    assert len(all_val) == len(set(all_val)), "duplicate validation case"

    OUT.write_text(json.dumps(
        {"seed": SEED, "n_folds": N_FOLDS, "n_cases": len(case_ids), "folds": folds},
        indent=2))
    print("\nEvery case appears in exactly one validation fold.")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
