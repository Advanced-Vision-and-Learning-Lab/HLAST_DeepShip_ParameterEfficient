# ───────────────────────────────────────────────────────────────
# File: demo_ESC50.py
# Description: ESC-50 experiment with structured logging & summary
# ───────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from Demo_Parameters import Parameters
from Utils.LitModel import LitModel
from Datasets.ESC50DataModule import ESC50DataModule

# ---------------------------------------------------------------
# Epoch-end printout
# ---------------------------------------------------------------
class EpochAccPrinter(Callback):
    def on_train_epoch_end(self, trainer, *_):
        t = trainer.callback_metrics.get("train_acc")
        v = trainer.callback_metrics.get("val_acc")
        if t is not None and v is not None:
            print(f"Epoch {trainer.current_epoch:03d} | train_acc={t:.4f} | val_acc={v:.4f}")

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("ESC-50 experiment")
    # core
    p.add_argument("--train_mode", default="histogram")
    p.add_argument("--num_epochs", default=200, type=int)
    p.add_argument("--train_batch_size", default=64, type=int)
    p.add_argument("--val_batch_size",   default=64, type=int)
    p.add_argument("--test_batch_size",  default=64, type=int)  # kept for Parameters()
    p.add_argument("--num_workers",      default=4,  type=int)
    # feature
    p.add_argument("--audio_feature", default="LogMelFBank")
    p.add_argument("--sample_rate",   default=16000, type=int)
    p.add_argument("--segment_length", default=5, type=int)
    p.add_argument("--window_length", default=2048, type=int)
    p.add_argument("--hop_length",    default=512, type=int)
    p.add_argument("--number_mels",   default=128, type=int)
    # histogram / adapter
    p.add_argument("--numBins", default=16, type=int)
    p.add_argument("--RR",      default=64, type=int)
    p.add_argument("--histograms_shared", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--histogram_location", default="mhsa")
    p.add_argument("--histogram_mode",     default="parallel")
    p.add_argument("--adapters_shared",    action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adapter_location",   default="ffn")
    p.add_argument("--adapter_mode",       default="parallel")
    # misc for Parameters
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--patience", default=25, type=int)
    p.add_argument("--model",   default="AST")
    p.add_argument("--lora_shared", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ssf_shared",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data_selection", default=3, type=int)
    p.add_argument("--use_pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_target", default="q")
    p.add_argument("--lora_rank",   default=6, type=int)
    p.add_argument("--bias_mode",   default="full")
    p.add_argument("--ssf_mode",    default="full")
    return p.parse_args()

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def experiment_name(params):
    return (f"{params['train_mode']}_{params['feature']}_wl{params['window_length']}_"
            f"hl{params['hop_length']}_m{params['number_mels']}")

# ---------------------------------------------------------------
# Single seed on single fold
# ---------------------------------------------------------------
def run_once(params, fold: int, run: int, dm: ESC50DataModule, log_root: str):
    seed_everything(42 + run, workers=True)

    lit_model = LitModel(params, params["Model_name"],
                         num_classes=50, numBins=params["numBins"], RR=params["RR"])

    logger = TensorBoardLogger(
        save_dir=os.path.join(log_root, f"Fold{fold}", f"Run{run}"),
        name="metrics"
    )
    ckpt  = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    trainer = L.Trainer(
        max_epochs=params["num_epochs"],
        logger=logger,
        callbacks=[ckpt, early, EpochAccPrinter()],
        log_every_n_steps=20,
        enable_progress_bar=False,
        accelerator="gpu",
        devices="auto",
    )
    trainer.fit(lit_model, datamodule=dm)
    return ckpt.best_model_score.item()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    args = parse_args()
    params = Parameters(args)

    # Build experiment root path
    root_log = os.path.join("tb_logs_esc50", experiment_name(params))
    os.makedirs(root_log, exist_ok=True)

    # Param count
    temp_model = LitModel(params, params["Model_name"], 50,
                          numBins=params["numBins"], RR=params["RR"])
    n_params = count_params(temp_model)
    print(f"Trainable parameters: {n_params:,}")

    dm = ESC50DataModule(
        batch_size={"train": args.train_batch_size, "val": args.val_batch_size},
        num_workers=args.num_workers,
    )
    dm.prepare_data()

    scores = []
    for fold in range(5):
        dm.set_fold(fold)
        dm.setup()
        for run in range(3):
            score = run_once(params, fold, run, dm, root_log)
            scores.append(score)

    mean_acc = np.mean(scores)
    std_acc  = np.std(scores)

    # Console summary
    print("\n──────── ESC-50 SUMMARY ────────")
    print(f"mean ± std : {mean_acc:.4f} ± {std_acc:.4f}")

    # Save summary txt
    summary_path = os.path.join(root_log, "summary.txt")
    summary = {
        "experiment_name": experiment_name(params),
        "train_mode": params["train_mode"],
        "trainable_params": int(n_params),
        "mean_accuracy": float(mean_acc),
        "std_accuracy":  float(std_acc),
        "settings": {k: params[k] for k in (
            "window_length", "hop_length", "number_mels",
            "numBins", "RR", "histogram_location", "histogram_mode",
            "adapter_location", "adapter_mode",
            "lora_target", "lora_rank", "bias_mode", "ssf_mode",
            "histograms_shared", "adapters_shared",
            "lora_shared", "ssf_shared"
        )}
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

