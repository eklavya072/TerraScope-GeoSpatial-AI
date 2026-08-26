"""Split-driven EuroSAT loading. Every fold comes from the committed split file.

There is exactly one way to obtain data in this project: read the committed
split CSV. No script is allowed to invent a fold, resample, or fall back to a
third-party split -- that is what makes each reported number traceable to a
file a reader can inspect.
"""

import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bench.config import CLASSES, CORPUS_ROOT, INPUT_SIZE, NORM_MEAN, NORM_STD, SPLIT_CSV

CACHE = os.path.join("data", "cache")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def read_split(split_csv: str = SPLIT_CSV) -> dict[str, list[tuple[str, int]]]:
    """fold -> [(relative_path, label_index), ...] in committed file order."""
    if not os.path.exists(split_csv):
        raise SystemExit(f"missing {split_csv} -- run scripts/make_split.py")
    folds: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    with open(split_csv) as fh:
        for row in csv.DictReader(fh):
            folds[row["fold"]].append((row["filename"], CLASS_TO_IDX[row["label"]]))
    return folds


def _load_fold(fold: str, items: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Decode a fold into a uint8 array, cached so epochs never re-decode JPEGs.

    27,000 64x64 tiles are only ~330 MB as uint8, so the whole corpus lives in
    RAM. Keeping decode out of the training loop also keeps the loop's cost
    dominated by the model, which matters when we later attribute energy.
    """
    os.makedirs(CACHE, exist_ok=True)
    npz = os.path.join(CACHE, f"{fold}_{len(items)}.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        return z["x"], z["y"]
    x = np.zeros((len(items), 64, 64, 3), dtype=np.uint8)
    y = np.zeros(len(items), dtype=np.int64)
    for i, (rel, label) in enumerate(items):
        with Image.open(os.path.join(CORPUS_ROOT, rel)) as im:
            x[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        y[i] = label
    np.savez(npz, x=x, y=y)
    return x, y


class EuroSATFold(Dataset):
    """One fold of the committed split.

    Augmentation is the dihedral group of flips and 90-degree rotations. For
    nadir satellite tiles these are genuinely label-preserving (unlike, say,
    colour jitter, whose safety depends on the sensor's radiometry), which is
    why they can be applied identically to every architecture without
    advantaging any of them.
    """

    def __init__(self, fold: str, split_csv: str = SPLIT_CSV, augment: bool = False,
                 seed: int = 0):
        items = read_split(split_csv)[fold]
        self.x, self.y = _load_fold(fold, items)
        self.augment = augment
        self.fold = fold
        self._g = torch.Generator().manual_seed(seed * 100_003 + hash(fold) % 100_003)
        self._mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
        self._std = torch.tensor(NORM_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = torch.from_numpy(self.x[i].copy()).permute(2, 0, 1).float().div_(255.0)
        if self.augment:
            r = torch.randint(0, 8, (1,), generator=self._g).item()
            if r & 1:
                img = torch.flip(img, dims=[2])          # horizontal
            if r & 2:
                img = torch.flip(img, dims=[1])          # vertical
            if r & 4:
                img = torch.rot90(img, 1, dims=[1, 2])   # 90 degrees
        img = (img - self._mean) / self._std
        if img.shape[-1] != INPUT_SIZE:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear", align_corners=False).squeeze(0)
        return img, int(self.y[i])
