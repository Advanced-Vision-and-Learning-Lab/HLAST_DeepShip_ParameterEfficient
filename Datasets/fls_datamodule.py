# fls_datamodule.py

import os
import math
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import lightning as L
from typing import Optional

class _ArrayDataset(Dataset):
    """
    Simple Dataset over in-memory numpy arrays (x: (N,H,W,1) float32 in [0,1], y: (N,))
    Applies a torchvision transform pipeline that expects CHW torch tensors.
    """
    def __init__(self, x_np: np.ndarray, y_np: np.ndarray, transform=None):
        assert x_np.ndim == 4 and x_np.shape[-1] in (1, 3), "Expected (N,H,W,C)"
        self.x = x_np
        self.y = y_np.astype(np.int64)  # torch CE expects Long
        self.transform = transform

        # ToTensor will convert HWC float32 -> CHW torch.float32 (no rescale since already 0..1 float)
        self._to_tensor = T.ToTensor()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]              # (H,W,1) float32 in [0,1]
        lbl = int(self.y[idx])
        # ToTensor expects HWC
        img_chw = self._to_tensor(img)  # (C,H,W)
        if self.transform is not None:
            img_chw = self.transform(img_chw)
        return img_chw, torch.tensor(lbl, dtype=torch.long)


def _load_hdf5(path):
    """
    Load x/y arrays from an HDF5 classification file.
    Returns dict with keys: x_train, y_train, x_val (opt), y_val (opt), x_test, y_test, class_names (list[str])
    """
    out = {}
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        # required
        out["x_train"] = f["x_train"][...].astype(np.float32)
        out["y_train"] = f["y_train"][...]
        out["x_test"]  = f["x_test"][...].astype(np.float32)
        out["y_test"]  = f["y_test"][...]
        # optional val
        if "x_val" in keys and "y_val" in keys:
            out["x_val"] = f["x_val"][...].astype(np.float32)
            out["y_val"] = f["y_val"][...]
        else:
            out["x_val"] = None
            out["y_val"] = None
        # class names
        if "class_names" in keys:
            cn = f["class_names"][...]
            out["class_names"] = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in cn]
        else:
            out["class_names"] = None
    return out


def _deterministic_stratified_val_split(x, y, val_ratio=0.10):
    """
    Create a deterministic stratified split by *index order* (no randomness).
    For each class k, take the FIRST ceil(0.10 * count_k) indices (by existing order) as val.
    Returns: x_train_new, y_train_new, x_val_new, y_val_new
    """
    y = y.astype(np.int64)
    classes = np.unique(y)
    train_idx = []
    val_idx = []
    for k in classes:
        idx_k = np.nonzero(y == k)[0]
        # keep current order
        n_k = len(idx_k)
        n_val = max(1, math.ceil(val_ratio * n_k))
        val_idx.extend(idx_k[:n_val])
        train_idx.extend(idx_k[n_val:])

    # keep global order within each split
    val_idx = np.array(sorted(val_idx))
    train_idx = np.array(sorted(train_idx))

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


class FLSDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,                 # "watertank" or "turntable"
        data_root: str = "./FLS",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        filename_watertank: str = "marine-debris-watertank-classification-96x96.hdf5",
        filename_turntable: str = "marine-debris-turntable-classification-object_classes-platform-96x96.hdf5",
    ):
        super().__init__()
        ds = dataset.lower()
        assert ds in {"watertank", "turntable"}, "dataset must be 'watertank' or 'turntable'"
        self.dataset = ds
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
        self.filename_watertank = filename_watertank
        self.filename_turntable = filename_turntable

        # to be filled in setup()
        self.train_mean = None
        self.class_names = None

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

        self.num_samples = {"train": 0, "val": 0, "test": 0}

    def prepare_data(self):
        # no downloads; verify files exist
        path = self._dataset_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HDF5 not found: {path}")

    def setup(self, stage: Optional[str] = None):
        path = self._dataset_path()
        arrays = _load_hdf5(path)
        self.class_names = arrays["class_names"]

        # Build splits
        if self.dataset == "watertank":
            x_tr, y_tr = arrays["x_train"], arrays["y_train"]
            x_va, y_va = arrays["x_val"], arrays["y_val"]  # provided by release
            x_te, y_te = arrays["x_test"], arrays["y_test"]
            if x_va is None or y_va is None:
                # Fallback: create deterministic 10% val if val isn't present
                x_tr, y_tr, x_va, y_va = _deterministic_stratified_val_split(x_tr, y_tr, val_ratio=0.10)
        else:  # turntable
            x_tr_full, y_tr_full = arrays["x_train"], arrays["y_train"]
            x_te, y_te = arrays["x_test"], arrays["y_test"]
            # Make deterministic 10% stratified val from the provided train set
            x_tr, y_tr, x_va, y_va = _deterministic_stratified_val_split(x_tr_full, y_tr_full, val_ratio=0.10)

        # Compute training mean AFTER final train split is set
        self.train_mean = float(x_tr.mean())  # scalar in [0,1], e.g., ~0.331 or ~0.397

        # Define transforms
        # Note: ToTensor() is handled inside the Dataset class; here we expect CHW tensors.
        train_tfms = T.Compose([
            # Augmentations (train only)
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.10, 0.10), fill=0),
            # Mean subtraction (std=1.0 -> just subtract mean)
            T.Normalize(mean=[self.train_mean], std=[1.0]),
        ])
        eval_tfms = T.Compose([
            T.Normalize(mean=[self.train_mean], std=[1.0]),
        ])

        # Build torch Datasets
        self.train_ds = _ArrayDataset(x_tr, y_tr, transform=train_tfms)
        self.val_ds   = _ArrayDataset(x_va, y_va, transform=eval_tfms)
        self.test_ds  = _ArrayDataset(x_te, y_te, transform=eval_tfms)


        n_tr = len(self.train_ds)
        n_va = len(self.val_ds)
        n_te = len(self.test_ds)
        
        self.num_samples.update({"train": n_tr, "val": n_va, "test": n_te})
        
        print(
            f"\n[FLSDataModule] dataset='{self.dataset}' | "
            f"train={n_tr} | val={n_va} | test={n_te} | "
            f"train_mean={self.train_mean:.3f}"
        )
        if self.class_names:
            print(f"[FLSDataModule] #classes={len(self.class_names)} -> {self.class_names}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,         
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    # ----- helpers -----
    def _dataset_path(self) -> str:
        if self.dataset == "watertank":
            fname = self.filename_watertank
        else:
            fname = self.filename_turntable
        return os.path.join(self.data_root, fname)


