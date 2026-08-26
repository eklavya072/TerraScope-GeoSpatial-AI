"""Train one (model, seed) pair under the single shared recipe.

Fairness is enforced structurally: the recipe comes from bench.config.RECIPE and
this script exposes no knobs that could differentiate one architecture from
another. The only command-line arguments are WHICH model and WHICH seed.

Results append one JSON row per run to results/runs.jsonl, each carrying the
recipe hash, the split hash, and the full environment fingerprint.

Usage:
    python -m bench.train --model resnet50 --seed 0
    python -m bench.train --model all --seeds 0,1,2,3,4
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from bench import utils
from bench.config import (CHECKPOINT_DIR, MODEL_ZOO, RECIPE, RESULTS_DIR,
                          SPLIT_CSV, recipe_hash)
from bench.data import EuroSATFold
from bench.models import build, param_count

RUNS_JSONL = os.path.join(RESULTS_DIR, "runs.jsonl")


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(epoch: int) -> float:
    """Linear warmup then cosine decay -- identical for every model."""
    warm = RECIPE["warmup_epochs"]
    total = RECIPE["max_epochs"]
    if epoch < warm:
        return (epoch + 1) / warm
    progress = (epoch - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_one(name: str, seed: int, device: torch.device, split_csv: str) -> dict:
    utils.set_seed(seed)
    split_sha = utils.verify_split(split_csv)

    bs = RECIPE["batch_size"]
    gen = torch.Generator().manual_seed(seed)
    tr = DataLoader(EuroSATFold("train", split_csv, augment=True, seed=seed),
                    batch_size=bs, shuffle=True, generator=gen, drop_last=False)
    va = DataLoader(EuroSATFold("val", split_csv), batch_size=256)
    te = DataLoader(EuroSATFold("test", split_csv), batch_size=256)

    model = build(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                            weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=RECIPE["label_smoothing"])

    best_val, best_epoch, bad, best_state = -1.0, -1, 0, None
    history, t0 = [], time.time()

    for epoch in range(RECIPE["max_epochs"]):
        model.train()
        te0, run_loss, nb = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1
        sched.step()
        val_acc = evaluate(model, va, device)
        history.append({"epoch": epoch, "train_loss": run_loss / nb,
                        "val_acc": val_acc, "seconds": round(time.time() - te0, 1)})
        print(f"  [{name} s{seed}] epoch {epoch:2d}  loss {run_loss/nb:.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time()-te0:.0f}s)", flush=True)

        if val_acc > best_val:
            best_val, best_epoch, bad = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= RECIPE["early_stopping"]["patience"]:
                print(f"  [{name} s{seed}] early stop at epoch {epoch} "
                      f"(best epoch {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)          # restore_best_weights
    test_acc = evaluate(model, te, device)
    elapsed = time.time() - t0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINT_DIR, f"{name}_seed{seed}.pt")
    torch.save({"model": name, "seed": seed, "state_dict": best_state,
                "recipe_hash": recipe_hash(), "split_sha256": split_sha}, ckpt)

    row = {
        "kind": "train",
        "model": name,
        "timm_id": MODEL_ZOO[name],
        "seed": seed,
        "params": param_count(model),
        "val_acc": best_val,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(elapsed, 1),
        "train_device": str(device),
        "checkpoint": ckpt,
        "split_csv": split_csv,
        "split_sha256": split_sha,
        "recipe_hash": recipe_hash(),
        "recipe": RECIPE,
        "history": history,
        "env": utils.environment(),
    }
    utils.append_jsonl(RUNS_JSONL, row)
    print(f"RESULT model={name} seed={seed} val_acc={best_val:.4f} "
          f"test_acc={test_acc:.4f} epochs={len(history)} "
          f"time={elapsed/60:.1f}min", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all",
                    help="zoo name, comma list, or 'all'")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--split", default=SPLIT_CSV)
    args = ap.parse_args()

    names = list(MODEL_ZOO) if args.model == "all" else args.model.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = pick_device(args.device)
    print(f"device={device}  recipe={recipe_hash()}  models={names}  seeds={seeds}",
          flush=True)

    for name in names:
        for seed in seeds:
            train_one(name, seed, device, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
