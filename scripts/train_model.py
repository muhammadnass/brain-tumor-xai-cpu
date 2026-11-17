"""
Train the 3D U-Net brain tumor segmentation model on real MRI data
(Medical Segmentation Decathlon Task01_BrainTumour).

File location: scripts/train_model.py

USAGE:
    # ALWAYS run this first to verify the pipeline works (takes ~2-5 min):
    python scripts/train_model.py --quick-test

    # Then run real training (CPU - expect this to be slow, hours not minutes):
    python scripts/train_model.py --epochs 20 --patch-size 64

    # If you hit memory errors, reduce patch size further:
    python scripts/train_model.py --epochs 20 --patch-size 48
"""
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_models.segmentation import create_segmentation_model
from config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "brats_dataset"
CHECKPOINT_PATH = MODELS_DIR / "segmentation" / "unet_3d.pth"


def remap_labels(label):
    """
    Decathlon Task01_BrainTumour labels: 0=background, 1=edema,
    2=non-enhancing tumor, 3=enhancing tumor.

    Our model expects: 0=background, 1=tumor, 2=edema.
    Merge classes 2+3 -> tumor (1), keep 1 -> edema (2).
    """
    import numpy as np
    remapped = np.zeros_like(label)
    remapped[label == 1] = 2   # edema -> class 2
    remapped[label == 2] = 1   # non-enhancing tumor -> class 1 (tumor)
    remapped[label == 3] = 1   # enhancing tumor -> class 1 (tumor)
    return remapped


SUBSET_DIR = DATA_DIR / "Task01_BrainTumour_subset"


def build_file_list():
    """
    Build a list of {image, label} file paths from our manually-extracted
    subset folder (NOT using monai.apps.DecathlonDataset, which expects
    the FULL 484-case dataset structure with a complete dataset.json -
    that breaks when we've only extracted a subset for disk space reasons).
    """
    images_dir = SUBSET_DIR / "imagesTr"
    labels_dir = SUBSET_DIR / "labelsTr"

    if not images_dir.exists():
        logger.error(f"❌ Subset not found at {images_dir}")
        logger.error("   Run this first: python scripts/extract_subset_from_tar.py")
        sys.exit(1)

    image_files = sorted(images_dir.glob("*.nii.gz"))
    file_list = []
    for img_path in image_files:
        label_path = labels_dir / img_path.name
        if label_path.exists():
            file_list.append({"image": str(img_path), "label": str(label_path)})

    if not file_list:
        logger.error(f"❌ No matching image/label pairs found in {SUBSET_DIR}")
        sys.exit(1)

    logger.info(f"✅ Found {len(file_list)} usable cases in subset")
    return file_list


def build_datasets(patch_size: int, max_cases: int = None):
    """Build MONAI training and validation datasets with transforms,
    using our manually-extracted subset (disk-space-friendly)."""
    from monai.data import Dataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        NormalizeIntensityd, RandCropByPosNegLabeld, RandFlipd,
        RandRotate90d, Lambdad, EnsureTyped,
    )

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=remap_labels),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # NOTE: pos:neg adjusted from 2:1 to 1:2 (moved toward more background
        # exposure) after diagnosing severe false-positive over-segmentation
        # on full-volume inference (see scripts/quick_case_evaluation.py findings:
        # Dice 0.077 on full volume vs 0.365 on tumor-biased patches). This is a
        # modest correction, not a complete fix - the primary lever remains
        # increased training data volume.
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=[patch_size, patch_size, patch_size],
            pos=1, neg=2, num_samples=2,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])

    # VALIDATION transforms deliberately use much heavier background sampling
    # (pos:neg = 1:5, ~17% foreground crops) to approximate the TRUE class
    # distribution seen at full-volume inference time (~0.65% tumor prevalence
    # in practice, though 1:5 patches are still far more foreground-rich than
    # that - a genuine full-volume validation would be even more representative
    # but is computationally expensive per-epoch on CPU; this is a practical
    # middle ground that at least reduces the previous 2:1 optimism bias).
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["label"], func=remap_labels),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=[patch_size, patch_size, patch_size],
            pos=1, neg=5, num_samples=1,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    file_list = build_file_list()
    if max_cases:
        file_list = file_list[:max_cases]

    # Simple 80/20 train/val split of our subset
    split_idx = max(1, int(len(file_list) * 0.8))
    train_files = file_list[:split_idx]
    val_files = file_list[split_idx:] if len(file_list) > split_idx else file_list[-1:]

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    logger.info(f"✅ Training cases: {len(train_ds)} | Validation cases: {len(val_ds)}")
    return train_ds, val_ds


def train(args):
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.data.utils import list_data_collate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    if not SUBSET_DIR.exists() or not (SUBSET_DIR / "imagesTr").exists():
        logger.error(f"❌ Dataset subset not found at {SUBSET_DIR}")
        logger.error("   Run this first: python scripts/extract_subset_from_tar.py")
        sys.exit(1)

    max_cases = 2 if args.quick_test else args.max_cases
    epochs = 1 if args.quick_test else args.epochs
    patch_size = 32 if args.quick_test else args.patch_size

    if args.quick_test:
        logger.info("=" * 60)
        logger.info("🧪 QUICK TEST MODE: 1 epoch, 2 cases, small patches")
        logger.info("   This just verifies the training loop works end-to-end.")
        logger.info("=" * 60)

    train_ds, val_ds = build_datasets(patch_size=patch_size, max_cases=max_cases)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=list_data_collate,
    )

    model = create_segmentation_model(device=str(device))
    model.train()

    loss_function = DiceCELoss(to_onehot_y=True, softmax=False, include_background=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training: {epochs} epoch(s), patch size {patch_size}³, "
                f"batch size {args.batch_size}")
    logger.info("⏱️  On CPU, each epoch may take several minutes to hours depending "
                "on dataset size and patch size. Progress prints below.")

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # already softmaxed
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            logger.info(f"  Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} "
                        f"| Loss: {loss.item():.4f}")

        epoch_loss /= max(step, 1)
        epoch_time = time.time() - epoch_start
        logger.info(f"✅ Epoch {epoch+1}/{epochs} complete | Avg Loss: {epoch_loss:.4f} "
                    f"| Time: {epoch_time:.1f}s")

        # ---- Validation ----
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_outputs = model(val_inputs)

                val_outputs_discrete = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_labels_onehot = torch.zeros_like(val_outputs)
                val_labels_onehot.scatter_(1, val_labels.long(), 1)
                val_outputs_onehot = torch.zeros_like(val_outputs)
                val_outputs_onehot.scatter_(1, val_outputs_discrete, 1)

                dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

        mean_dice = dice_metric.aggregate().item()
        logger.info(f"📊 Validation Dice Score: {mean_dice:.4f}")
        model.train()

        # Save checkpoint if improved
        if mean_dice > best_dice:
            best_dice = mean_dice
            model.save_checkpoint(
                str(CHECKPOINT_PATH),
                metadata={'epoch': epoch + 1, 'dice_score': best_dice}
            )
            logger.info(f"💾 New best model saved (Dice: {best_dice:.4f}) -> {CHECKPOINT_PATH}")

    logger.info("=" * 60)
    logger.info(f"✅ TRAINING COMPLETE | Best Dice Score: {best_dice:.4f}")
    logger.info(f"   Checkpoint saved to: {CHECKPOINT_PATH}")
    logger.info("=" * 60)

    if args.quick_test:
        logger.info("")
        logger.info("🎉 Quick test passed! The training loop works correctly.")
        logger.info("   Now run real training, e.g.:")
        logger.info("   python scripts/train_model.py --epochs 20 --patch-size 64")


def main():
    parser = argparse.ArgumentParser(description="Train brain tumor segmentation model")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run a tiny 1-epoch test on 2 cases to verify the pipeline works")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patch-size", type=int, default=64,
                        help="3D patch size (cube side length). Reduce if you hit memory errors.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of training cases (useful for faster iteration)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
