# =================================================================
# HISTOPATHOLOGY DATA MODULE  (Experiment 2: cross-modality)
# Source domain: PatchCamelyon (PCam), via the Kaggle "Histopathologic
#   Cancer Detection" competition (a de-duplicated PCam release: 96x96
#   H&E patches, binary tumor/normal, /kaggle/input/histopathologic-
#   cancer-detection/train/*.tif + train_labels.csv).
# Target domain: BreaKHis (breast tumor histopathology, single-lab
#   collection -- P&D Laboratory, Brazil), used in place of WILDS
#   Camelyon17 specifically to avoid the ~10GB WILDS download; see
#   BreakHisTargetPool below for the honest tradeoff this involves.
#   WildsCamelyon17TargetPool is KEPT (not deleted) as a documented
#   fallback for anyone with the bandwidth/time budget for the full
#   multi-hospital dataset.
#
# The actual shift mechanism (Fourier Domain Adaptation, low-frequency
# amplitude blending) is UNCHANGED from Experiment 1 -- imported directly
# from cross_domain_shift.py. Only the data source/target and image
# preprocessing (RGB, no grayscale conversion) differ.
# =================================================================

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # fallback base class so PCamDataset can still be defined

try:
    from wilds import get_dataset as wilds_get_dataset
    _WILDS_AVAILABLE = True
except ImportError:
    _WILDS_AVAILABLE = False


# =================================================================
# 1. PCAM SOURCE DOMAIN (D_s)
# =================================================================
class PCamDataset(Dataset):
    """
    Wraps the Kaggle "Histopathologic Cancer Detection" release of PCam:
    a directory of .tif patches + a labels CSV (id, label). The Kaggle
    competition only ships a labeled TRAIN split (its "test" split is
    unlabeled, held out for submission scoring) -- so this class expects
    a pre-split list of (path, label) pairs, produced by
    `build_pcam_splits()` below, rather than doing the split itself.
    """
    def __init__(self, records, transform):
        self.records = records  # list of (filepath, label) tuples
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def build_pcam_splits(root="/kaggle/input/histopathologic-cancer-detection",
                       val_frac=0.15, test_frac=0.15, seed=42):
    """
    Reads train_labels.csv and produces a fixed, seeded, stratified
    train/val/test split (Kaggle's own "test" directory is unlabeled and
    NOT used here). Stratification keeps the tumor/normal class ratio
    consistent across splits.
    """
    labels_path = os.path.join(root, "train_labels.csv")
    img_dir = os.path.join(root, "train")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"PCam labels not found at {labels_path}. Add the 'Histopathologic "
            f"Cancer Detection' competition dataset as a Kaggle notebook input, "
            f"or pass a custom `root`."
        )

    df = pd.read_csv(labels_path)
    df["path"] = df["id"].apply(lambda x: os.path.join(img_dir, f"{x}.tif"))

    rng = np.random.RandomState(seed)
    records_by_class = {c: df[df["label"] == c][["path", "label"]].values.tolist()
                         for c in df["label"].unique()}

    train_records, val_records, test_records = [], [], []
    for c, recs in records_by_class.items():
        rng.shuffle(recs)
        n = len(recs)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val_records += recs[:n_val]
        test_records += recs[n_val:n_val + n_test]
        train_records += recs[n_val + n_test:]

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def get_pcam_transform():
    """RGB microscopy patches -- resize to 224 (native PCam is 96x96),
    NO grayscale conversion (unlike the chest X-ray pipeline), standard
    ImageNet normalization for the transfer-learned backbones."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_pcam_data(root="/kaggle/input/histopathologic-cancer-detection", seed=42):
    """Returns (train_ds, val_ds, test_ds, num_classes=2)."""
    transform = get_pcam_transform()
    train_records, val_records, test_records = build_pcam_splits(root=root, seed=seed)
    train_ds = PCamDataset(train_records, transform)
    val_ds = PCamDataset(val_records, transform)
    test_ds = PCamDataset(test_records, transform)
    return train_ds, val_ds, test_ds, 2


# =================================================================
# 2. WILDS CAMELYON17 TARGET DOMAIN POOL  (fallback -- ~10GB download)
# =================================================================
class WildsCamelyon17TargetPool:
    """
    Loads a fixed, seeded pool of WILDS Camelyon17 patches (real,
    multi-hospital tumor/normal patches) to serve as FDA amplitude-
    blending partners, mirroring RSNATargetPool's role in Experiment 1.

    KEPT AS A FALLBACK: this is the strongest option in terms of shift
    realism (genuine, explicitly-labeled 5-hospital covariate shift, same
    binary task, same patch format as PCam) but requires a ~10GB one-time
    download via the `wilds` package, which was judged too costly for a
    weekly iteration cycle. See BreakHisTargetPool below for the
    lightweight default used instead, and its documented tradeoff.

    Requires: pip install wilds
    First call triggers a real download (~10GB) unless already cached
    under `wilds_root`.
    """
    def __init__(self, wilds_root="/kaggle/working/wilds_data", pool_size=500, seed=42):
        if not _WILDS_AVAILABLE:
            raise ImportError("The 'wilds' package is required for the target domain. "
                               "Run: pip install wilds -q")

        dataset = wilds_get_dataset(dataset="camelyon17", download=True, root_dir=wilds_root)
        # Use the full metadata array to sample across all hospitals/centers,
        # not just one split, since the target pool's role is to supply
        # REAL cross-institution style variation, not to serve as its own
        # labeled evaluation set.
        n_total = len(dataset)
        rng = np.random.RandomState(seed)
        self.indices = rng.choice(n_total, size=min(pool_size, n_total), replace=False)
        self.dataset = dataset
        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._cache = {}

    def __len__(self):
        return len(self.indices)

    def get(self, i):
        """Returns a (3, 224, 224) tensor in [0,1], cached after first read."""
        if i in self._cache:
            return self._cache[i]
        img, _, _ = self.dataset[self.indices[i]]  # PIL image, label, metadata
        tensor = self.resize(img.convert("RGB"))
        self._cache[i] = tensor
        return tensor

    def preload_all(self):
        return torch.stack([self.get(i) for i in range(len(self.indices))])


# =================================================================
# 2b. BREAKHIS TARGET DOMAIN POOL  (default -- lightweight, ~1-2GB)
# =================================================================
class BreakHisTargetPool:
    """
    Loads a fixed, seeded pool of BreaKHis breast-tumor histopathology
    images (Spanhol et al., 2016) to serve as FDA amplitude-blending
    partners, in place of WildsCamelyon17TargetPool.

    WHY THIS INSTEAD OF WILDS CAMELYON17: BreaKHis is roughly 1-2GB
    (several Kaggle mirrors are under 1GB) versus WILDS Camelyon17's
    ~10GB one-time download -- a meaningful difference for a weekly
    Kaggle iteration cycle.

    HONEST TRADEOFF, stated plainly: BreaKHis was collected at a SINGLE
    lab (P&D Laboratory, Parana, Brazil), not across multiple named
    clinical institutions like WILDS Camelyon17's explicit 5-hospital
    design. What this pool provides is a real, independently-collected
    breast-tumor histopathology dataset with genuine differences in
    staining, acquisition, and patient population from PCam -- a
    meaningful covariate shift, but a weaker "multi-institution" story
    than the WILDS fallback above. Worth a sentence of honest framing in
    the paper: this tests generalization across an independently
    collected dataset and acquisition pipeline, not named clinical sites.

    ROBUST TO DIFFERENT KAGGLE MIRROR LAYOUTS: rather than assuming one
    specific folder structure (mirrors vary -- some nest by benign/
    malignant, some by magnification level, some flatten everything),
    this scans recursively under `root` for image files and uses whatever
    it finds. Labels are NOT needed here (unlike PCam's source-domain
    role) since the target pool only supplies partner IMAGES for the FDA
    blend, never its own labels.
    """
    def __init__(self, root="/kaggle/input/breakhis", pool_size=500, seed=42,
                 extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise FileNotFoundError(
                f"No image files found under {root}. Add a BreaKHis dataset as a "
                f"Kaggle notebook input (e.g. search 'breakhis' in Kaggle Datasets), "
                f"check the actual mount path with `!ls /kaggle/input/`, and pass it "
                f"as `root` if it differs from the default."
            )

        rng = np.random.RandomState(seed)
        n = min(pool_size, len(image_paths))
        chosen = rng.choice(len(image_paths), size=n, replace=False)
        self.paths = [image_paths[i] for i in chosen]

        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def get(self, i):
        """Returns a (3, 224, 224) tensor in [0,1], cached after first read."""
        if i in self._cache:
            return self._cache[i]
        img = Image.open(self.paths[i]).convert("RGB")
        tensor = self.resize(img)
        self._cache[i] = tensor
        return tensor

    def preload_all(self):
        return torch.stack([self.get(i) for i in range(len(self.paths))])
