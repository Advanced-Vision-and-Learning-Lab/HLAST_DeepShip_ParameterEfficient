# ───────────────────────────────────────────────────────────────
# File: Datasets/ESC50DataModule.py       ← replace entire file
# ───────────────────────────────────────────────────────────────
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import lightning as L


# ---------------------------------------------------------------
# Resample helper
# ---------------------------------------------------------------
def _verify_or_resample(src: Path, dst: Path, sr: int = 16_000, dur: float = 5.0):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        info = torchaudio.info(str(dst))
        ok = (
            info.sample_rate == sr
            and info.num_channels == 1
            and abs(info.num_frames - sr * dur) <= sr * 0.01
        )
        if ok:
            return
        dst.unlink()

    wav, in_sr = torchaudio.load(str(src))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    need = int(sr * dur)
    wav = wav[:, :need] if wav.shape[1] >= need else torch.nn.functional.pad(wav, (0, need - wav.shape[1]))
    torchaudio.save(str(dst), wav, sr, bits_per_sample=16)


def _prepare_esc50_16k(src_root: Path, out_root: Path, manifest: Path):
    if manifest.exists():
        return

    df = pd.read_csv(src_root / "meta" / "esc50.csv")
    class2idx = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
    items: List[Dict] = []

    for _, row in df.iterrows():
        rel = f"fold{row['fold']}/{row['category']}/{row['filename']}"
        _verify_or_resample(src_root / "audio" / row["filename"], out_root / rel)
        items.append(dict(relpath=rel, fold=int(row["fold"]), label=class2idx[row["category"]]))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(dict(classes=class2idx, items=items), f, indent=2)
    print("[ESC-50] 16 kHz dataset prepared.")


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class ESC50Dataset(Dataset):
    def __init__(self, items: List[Dict], root: Path):
        self.items, self.root = items, root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav, _ = torchaudio.load(str(self.root / it["relpath"]))
        return wav.squeeze(0), it["label"]


# ---------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------
class ESC50DataModule(L.LightningDataModule):
    """Train on 4 folds, validate on the held-out fold (folds 1-5)."""

    def __init__(
        self,
        esc50_root: str = "Datasets/ESC-50-master",
        processed_root: str = "Datasets/ESC50_16k",
        batch_size: Union[int, Dict[str, int]] = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.esc50_root = Path(esc50_root)
        self.proc_root = Path(processed_root)
        self.manifest_path = self.proc_root / "esc50_manifest.json"
        self.batch_size = (
            batch_size if isinstance(batch_size, dict) else {"train": batch_size, "val": batch_size}
        )
        self.num_workers = num_workers
        self._items: Optional[List[Dict]] = None
        self.fold_number = 1  # will hold 1-5

    # External control: loop index 0-4 → CSV fold 1-5
    def set_fold(self, idx_zero_based: int):
        self.fold_number = idx_zero_based + 1  # map 0→1, …, 4→5

    # Lightning hooks
    def prepare_data(self):
        _prepare_esc50_16k(self.esc50_root, self.proc_root, self.manifest_path)

    def setup(self, stage: Optional[str] = None):
        if self._items is None:
            with open(self.manifest_path) as f:
                self._items = json.load(f)["items"]

        train_items = [it for it in self._items if it["fold"] != self.fold_number]
        val_items   = [it for it in self._items if it["fold"] == self.fold_number]

        self.ds_train = ESC50Dataset(train_items, self.proc_root)
        self.ds_val   = ESC50Dataset(val_items,  self.proc_root)

    # DataLoaders
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

