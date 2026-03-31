"""
Dataset loading module for Breast Cancer Prediction.
Combines DDSM, BUS_UC, and BreakHis datasets into a unified binary classification dataset.

Anti-leakage strategy:
  - Patient/slide-level splitting (GroupShuffleSplit)
  - BreakHis: all magnifications of the same tissue stay in the same split
  - DDSM: all crops/views of the same case stay in the same split
  - Source tracking for per-dataset evaluation
"""

import os
import re
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


def _ddsm_case_id(filename):
    """Return the DDSM case identifier from a filename."""
    return filename.split('.')[0]


def _is_ddsm_augmented_variant(filename):
    """Detect DDSM generated variants like '... (2).png'."""
    stem = os.path.splitext(filename)[0]
    return bool(re.search(r"\(\d+\)$", stem))


def _collect_ddsm_entries(image_exts):
    """
    Collect DDSM image records with optional strict curation.

    Returns:
        entries: list of tuples (path, label, source, group_id, case_id)
        stats: dict with curated counts and removal details
    """
    entries = []
    removed_augmented = 0

    def scan_directory(directory, label):
        nonlocal removed_augmented
        if not os.path.isdir(directory):
            print(f"  [WARNING] Directory not found: {directory}")
            return

        for fname in os.listdir(directory):
            if os.path.splitext(fname)[1].lower() not in image_exts:
                continue

            if (
                getattr(config, "AUTHENTIC_EVAL", False)
                and getattr(config, "DDSM_EXCLUDE_AUGMENTED_VARIANTS", False)
                and _is_ddsm_augmented_variant(fname)
            ):
                removed_augmented += 1
                continue

            case_id = _ddsm_case_id(fname)
            entries.append(
                (os.path.join(directory, fname), label, "DDSM", f"ddsm_{case_id}", case_id)
            )

    scan_directory(config.DDSM_BENIGN, 0)
    scan_directory(config.DDSM_MALIGNANT, 1)

    removed_ambiguous = 0
    if (
        getattr(config, "AUTHENTIC_EVAL", False)
        and getattr(config, "DDSM_EXCLUDE_AMBIGUOUS_CASES", False)
    ):
        case_to_labels = {}
        for _, label, _, _, case_id in entries:
            case_to_labels.setdefault(case_id, set()).add(label)

        ambiguous_cases = {case_id for case_id, lbls in case_to_labels.items() if len(lbls) > 1}
        if ambiguous_cases:
            filtered = [record for record in entries if record[4] not in ambiguous_cases]
            removed_ambiguous = len(entries) - len(filtered)
            entries = filtered

    benign_count = sum(1 for _, label, _, _, _ in entries if label == 0)
    malignant_count = sum(1 for _, label, _, _, _ in entries if label == 1)

    stats = {
        "benign": benign_count,
        "malignant": malignant_count,
        "removed_augmented": removed_augmented,
        "removed_ambiguous": removed_ambiguous,
    }
    return entries, stats


def _extract_patient_id(filename, source_name):
    """
    Extract a patient/slide-level group ID from a filename.
    All images belonging to the same patient/slide will share the same group ID.
    This is critical for leak-free splitting.

    BreakHis format: SOB_B_A-14-22549AB-40-001.png
      - parts[1]+'-'+parts[2] = '14-22549AB' is the slide/patient ID
      - parts[3] = magnification (40, 100, 200, 400)
      - parts[4] = crop index
      -> group = 'breakhis_14-22549AB' (same across magnifications & crops)

    DDSM format: D1_A_1177_1.RIGHT_CC (2).png
      - text before first '.' is the case ID
      -> group = 'ddsm_D1_A_1177_1'

    BUS_UC format: 01.png, 010.png (simple numeric)
      - no patient grouping info available
      -> group = 'busuc_<filename>' (each image = its own group)
    """
    if source_name.startswith("BreakHis"):
        parts = filename.split('-')
        if len(parts) >= 3:
            return f"breakhis_{parts[1]}-{parts[2]}"
        return f"breakhis_{filename}"
    elif source_name == "DDSM":
        case_id = _ddsm_case_id(filename)
        return f"ddsm_{case_id}"
    elif source_name == "BUS_UC":
        return f"busuc_{filename}"
    return f"unknown_{filename}"


def collect_all_image_paths():
    """
    Collect all image paths from the three datasets with their labels.
    Returns:
        image_paths: list of file paths
        labels: list of int labels (0 = Benign, 1 = Malignant)
        sources: list of source dataset names
        group_ids: list of patient/slide-level group IDs for leak-free splitting
    """
    image_paths = []
    labels = []
    sources = []
    group_ids = []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def add_images(directory, label, source_name):
        if not os.path.isdir(directory):
            print(f"  [WARNING] Directory not found: {directory}")
            return 0
        count = 0
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in image_exts:
                image_paths.append(os.path.join(directory, f))
                labels.append(label)
                sources.append(source_name)
                group_ids.append(_extract_patient_id(f, source_name))
                count += 1
        return count

    print("=" * 60)
    print("Collecting images from all datasets...")
    print("=" * 60)

    # 1. DDSM Dataset (with optional strict curation)
    ddsm_entries, ddsm_stats = _collect_ddsm_entries(image_exts)
    for img_path, label, source, group_id, _case_id in ddsm_entries:
        image_paths.append(img_path)
        labels.append(label)
        sources.append(source)
        group_ids.append(group_id)

    print(f"  DDSM Benign:     {ddsm_stats['benign']} images")
    print(f"  DDSM Malignant:  {ddsm_stats['malignant']} images")
    if getattr(config, "AUTHENTIC_EVAL", False):
        if ddsm_stats["removed_augmented"] > 0:
            print(f"    [DDSM strict] Removed augmented variants: {ddsm_stats['removed_augmented']}")
        if ddsm_stats["removed_ambiguous"] > 0:
            print(f"    [DDSM strict] Removed ambiguous-case images: {ddsm_stats['removed_ambiguous']}")

    # 2. BUS_UC Dataset
    n = add_images(config.BUS_UC_BENIGN, 0, "BUS_UC")
    print(f"  BUS_UC Benign:   {n} images")
    n = add_images(config.BUS_UC_MALIGNANT, 1, "BUS_UC")
    print(f"  BUS_UC Malignant:{n} images")

    # 3. BreakHis Dataset (all magnifications)
    for mag in config.BREAKHIS_MAGNIFICATIONS:
        benign_dir = os.path.join(config.BREAKHIS_BASE, mag, "benign")
        malignant_dir = os.path.join(config.BREAKHIS_BASE, mag, "malignant")
        nb = add_images(benign_dir, 0, f"BreakHis_{mag}")
        nm = add_images(malignant_dir, 1, f"BreakHis_{mag}")
        print(f"  BreakHis {mag}: benign={nb}, malignant={nm}")

    total = len(image_paths)
    n_benign = labels.count(0)
    n_malignant = labels.count(1)
    n_groups = len(set(group_ids))
    print("-" * 60)
    print(f"  TOTAL: {total} images  (Benign: {n_benign}, Malignant: {n_malignant})")
    print(f"  Unique patient/slide groups: {n_groups}")
    print("=" * 60)

    return image_paths, labels, sources, group_ids


def get_transforms(is_training=True):
    """Get data transforms for training or evaluation."""
    if is_training:
        aug = config.AUGMENTATION
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if aug["vertical_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug["random_rotation"]),
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
                saturation=aug["color_jitter_saturation"],
                hue=aug["color_jitter_hue"],
            ),
            transforms.RandomAffine(
                degrees=aug["random_affine_degrees"],
                translate=aug["random_affine_translate"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=aug["random_erasing_prob"]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform


class BreastCancerDataset(Dataset):
    """Custom Dataset for breast cancer classification."""

    def __init__(self, image_paths, labels, transform=None, sources=None, group_ids=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.sources = sources  # optional: for per-dataset evaluation
        self.group_ids = group_ids  # optional: for case-level evaluation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a blank image on error so training doesn't crash
            print(f"  [WARNING] Could not load {img_path}: {e}")
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def _verify_no_group_leakage(train_groups, val_groups, test_groups):
    """Verify that no patient/slide group appears in more than one split."""
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    tv_leak = train_set & val_set
    tt_leak = train_set & test_set
    vt_leak = val_set & test_set

    if tv_leak or tt_leak or vt_leak:
        print("  [ERROR] DATA LEAKAGE DETECTED!")
        if tv_leak:
            print(f"    Train-Val overlap:  {len(tv_leak)} groups: {list(tv_leak)[:5]}")
        if tt_leak:
            print(f"    Train-Test overlap: {len(tt_leak)} groups: {list(tt_leak)[:5]}")
        if vt_leak:
            print(f"    Val-Test overlap:   {len(vt_leak)} groups: {list(vt_leak)[:5]}")
        raise ValueError("Data leakage: patient groups overlap between splits!")
    else:
        print("  [OK] No data leakage: all patient groups are split-exclusive.")
        print(f"    Train groups: {len(train_set)}, Val groups: {len(val_set)}, Test groups: {len(test_set)}")


def create_dataloaders():
    """
    Create train, validation, and test dataloaders.
    Uses PATIENT-LEVEL GroupShuffleSplit to prevent data leakage:
      - All images from the same patient/slide stay in the same split.
      - BreakHis: all magnifications & crops of a slide grouped together.
      - DDSM: all views & crops of a case grouped together.
    Uses weighted sampling to handle class imbalance.
    """
    image_paths, labels, sources, group_ids = collect_all_image_paths()

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    sources = np.array(sources)
    group_ids = np.array(group_ids)

    indices = np.arange(len(image_paths))

    # --- Split 1: train+val / test (at group level) ---
    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED)
    trainval_idx, test_idx = next(gss1.split(indices, labels, groups=group_ids))

    # --- Split 2: train / val (at group level) ---
    val_relative = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=config.RANDOM_SEED)
    train_idx_rel, val_idx_rel = next(gss2.split(
        trainval_idx, labels[trainval_idx], groups=group_ids[trainval_idx]
    ))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    X_train, y_train = image_paths[train_idx], labels[train_idx]
    X_val, y_val = image_paths[val_idx], labels[val_idx]
    X_test, y_test = image_paths[test_idx], labels[test_idx]
    s_train, s_val, s_test = sources[train_idx], sources[val_idx], sources[test_idx]
    g_train, g_val, g_test = group_ids[train_idx], group_ids[val_idx], group_ids[test_idx]

    # Verify no leakage
    _verify_no_group_leakage(g_train, g_val, g_test)

    print(f"\nSplit sizes (patient-level, leak-free):")
    print(f"  Train: {len(X_train)} (Benign: {(y_train==0).sum()}, Malignant: {(y_train==1).sum()})")
    print(f"  Val:   {len(X_val)}  (Benign: {(y_val==0).sum()}, Malignant: {(y_val==1).sum()})")
    print(f"  Test:  {len(X_test)}  (Benign: {(y_test==0).sum()}, Malignant: {(y_test==1).sum()})")

    # Show per-source split breakdown
    print(f"\n  Per-source breakdown:")
    for src in sorted(set(sources)):
        n_tr = (s_train == src).sum()
        n_va = (s_val == src).sum()
        n_te = (s_test == src).sum()
        print(f"    {src:.<25} train={n_tr:>5}, val={n_va:>5}, test={n_te:>5}")

    # Transforms
    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)

    # Datasets (pass sources for per-dataset evaluation on test set)
    train_dataset = BreastCancerDataset(
        X_train.tolist(), y_train.tolist(), train_transform, group_ids=g_train.tolist()
    )
    val_dataset = BreastCancerDataset(
        X_val.tolist(), y_val.tolist(), eval_transform, group_ids=g_val.tolist()
    )
    test_dataset = BreastCancerDataset(
        X_test.tolist(), y_test.tolist(), eval_transform,
        sources=s_test.tolist(), group_ids=g_test.tolist()
    )

    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


def collect_single_dataset_paths(dataset_name):
    """
    Collect image paths for a SINGLE dataset only.
    Supported dataset_name: 'DDSM', 'BUS_UC', 'BreakHis'

    Returns:
        image_paths, labels, sources, group_ids
    """
    image_paths = []
    labels = []
    sources = []
    group_ids = []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def add_images(directory, label, source_name):
        if not os.path.isdir(directory):
            print(f"  [WARNING] Directory not found: {directory}")
            return 0
        count = 0
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in image_exts:
                image_paths.append(os.path.join(directory, f))
                labels.append(label)
                sources.append(source_name)
                group_ids.append(_extract_patient_id(f, source_name))
                count += 1
        return count

    print("=" * 60)
    print(f"Collecting images from: {dataset_name}")
    print("=" * 60)

    if dataset_name == "DDSM":
        ddsm_entries, ddsm_stats = _collect_ddsm_entries(image_exts)
        for img_path, label, source, group_id, _case_id in ddsm_entries:
            image_paths.append(img_path)
            labels.append(label)
            sources.append(source)
            group_ids.append(group_id)

        print(f"  DDSM Benign:     {ddsm_stats['benign']} images")
        print(f"  DDSM Malignant:  {ddsm_stats['malignant']} images")
        if getattr(config, "AUTHENTIC_EVAL", False):
            if ddsm_stats["removed_augmented"] > 0:
                print(f"    [DDSM strict] Removed augmented variants: {ddsm_stats['removed_augmented']}")
            if ddsm_stats["removed_ambiguous"] > 0:
                print(f"    [DDSM strict] Removed ambiguous-case images: {ddsm_stats['removed_ambiguous']}")

    elif dataset_name == "BUS_UC":
        n = add_images(config.BUS_UC_BENIGN, 0, "BUS_UC")
        print(f"  BUS_UC Benign:   {n} images")
        n = add_images(config.BUS_UC_MALIGNANT, 1, "BUS_UC")
        print(f"  BUS_UC Malignant:{n} images")

    elif dataset_name == "BreakHis":
        for mag in config.BREAKHIS_MAGNIFICATIONS:
            benign_dir = os.path.join(config.BREAKHIS_BASE, mag, "benign")
            malignant_dir = os.path.join(config.BREAKHIS_BASE, mag, "malignant")
            nb = add_images(benign_dir, 0, f"BreakHis_{mag}")
            nm = add_images(malignant_dir, 1, f"BreakHis_{mag}")
            print(f"  BreakHis {mag}: benign={nb}, malignant={nm}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'DDSM', 'BUS_UC', or 'BreakHis'.")

    total = len(image_paths)
    n_benign = labels.count(0)
    n_malignant = labels.count(1)
    n_groups = len(set(group_ids))
    print("-" * 60)
    print(f"  TOTAL: {total} images  (Benign: {n_benign}, Malignant: {n_malignant})")
    print(f"  Unique patient/slide groups: {n_groups}")
    print("=" * 60)

    return image_paths, labels, sources, group_ids


def create_dataloaders_single(dataset_name):
    """
    Create train, validation, and test dataloaders for a SINGLE dataset.
    Uses the same patient-level GroupShuffleSplit and weighted sampling
    as the combined version, but restricted to one source dataset.

    Args:
        dataset_name: 'DDSM', 'BUS_UC', or 'BreakHis'

    Returns:
        train_loader, val_loader, test_loader
    """
    image_paths, labels, sources, group_ids = collect_single_dataset_paths(dataset_name)

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    sources = np.array(sources)
    group_ids = np.array(group_ids)

    indices = np.arange(len(image_paths))

    # --- Split 1: train+val / test (at group level) ---
    gss1 = GroupShuffleSplit(n_splits=1, test_size=config.TEST_RATIO, random_state=config.RANDOM_SEED)
    trainval_idx, test_idx = next(gss1.split(indices, labels, groups=group_ids))

    # --- Split 2: train / val (at group level) ---
    val_relative = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=config.RANDOM_SEED)
    train_idx_rel, val_idx_rel = next(gss2.split(
        trainval_idx, labels[trainval_idx], groups=group_ids[trainval_idx]
    ))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    X_train, y_train = image_paths[train_idx], labels[train_idx]
    X_val, y_val = image_paths[val_idx], labels[val_idx]
    X_test, y_test = image_paths[test_idx], labels[test_idx]
    s_test = sources[test_idx]
    g_train = group_ids[train_idx]
    g_val = group_ids[val_idx]
    g_test = group_ids[test_idx]

    # Verify no leakage
    _verify_no_group_leakage(g_train, g_val, g_test)

    print(f"\nSplit sizes for [{dataset_name}] (patient-level, leak-free):")
    print(f"  Train: {len(X_train)} (Benign: {(y_train==0).sum()}, Malignant: {(y_train==1).sum()})")
    print(f"  Val:   {len(X_val)}  (Benign: {(y_val==0).sum()}, Malignant: {(y_val==1).sum()})")
    print(f"  Test:  {len(X_test)}  (Benign: {(y_test==0).sum()}, Malignant: {(y_test==1).sum()})")

    # Transforms
    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)

    # Datasets
    train_dataset = BreastCancerDataset(
        X_train.tolist(), y_train.tolist(), train_transform, group_ids=g_train.tolist()
    )
    val_dataset = BreastCancerDataset(
        X_val.tolist(), y_val.tolist(), eval_transform, group_ids=g_val.tolist()
    )
    test_dataset = BreastCancerDataset(
        X_test.tolist(), y_test.tolist(), eval_transform,
        sources=s_test.tolist(), group_ids=g_test.tolist()
    )

    # Weighted sampler for class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Dataloaders — use smaller batch for small datasets
    batch_size = min(config.BATCH_SIZE, len(X_train) // 4) if len(X_train) < config.BATCH_SIZE * 4 else config.BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test — also verifies leak-free splitting
    train_loader, val_loader, test_loader = create_dataloaders()
    batch = next(iter(train_loader))
    print(f"\nSample batch: images={batch[0].shape}, labels={batch[1].shape}")
    print(f"Test dataset has source tracking: {test_loader.dataset.sources is not None}")
