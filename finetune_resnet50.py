"""Fine-tune the ResNet50 classifier by unfreezing the last stage.

The first ResNet50 pass (train_resnet50.py) froze the whole backbone and
trained only the head -> 93.6% EuroSAT test accuracy. This script takes the
next step: unfreeze the LAST ResNet50 stage (conv5 block) plus the head and
fine-tune at a low LR with the same augmentation as the original MobileNetV2
training (flip / brightness / contrast), saving models/resnet50_eurosat_ft.h5.

CPU note: the full forward pass is ~12 ms/img, so a fine-tune epoch over the
18,900-image train split is ~8-10 min here. The GPU path (Train_ResNet50_
Colab.ipynb) does the same unfreezing much faster and is the route to ~98%.

Usage:
    python finetune_resnet50.py
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from model_handler import resnet50_preprocess  # registers 'terra>resnet50_preprocess'

BATCH = 128
LR = 1e-4
MAX_EPOCHS = 8
EARLY_PATIENCE = 3
START = os.path.join('models', 'resnet50_eurosat.h5')
OUT = os.path.join('models', 'resnet50_eurosat_ft.h5')


def load_eurosat():
    from datasets import load_dataset
    hf = load_dataset('giswqs/EuroSAT_RGB')

    def _load(split):
        ds = hf[split]
        imgs = np.array([np.array(ds[i]['image'], dtype=np.float32) / 255.0
                         for i in range(len(ds))])
        return imgs, np.array([ds[i]['label'] for i in range(len(ds))])

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (
        _load(s) for s in ('train', 'validation', 'test'))
    print(f'train {x_tr.shape} | val {x_va.shape} | test {x_te.shape}', flush=True)
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def make_ds(x, y, batch, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(5000)
    if augment:
        # value_range is REQUIRED: without it this Keras version treats the
        # brightness/contrast factors as [0,255]-style and blows float [0,1]
        # images up to ~26 (measured), collapsing the model to near-random.
        aug = tf.keras.Sequential([
            layers.RandomFlip('horizontal_and_vertical'),
            layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
            layers.RandomContrast(0.1, value_range=(0.0, 1.0)),
        ])
        ds = ds.map(lambda img, lbl: (aug(img), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def main():
    print(f'[1/4] loading {START} ...', flush=True)
    model = tf.keras.models.load_model(START)

    # unfreeze the LAST ResNet50 stage (conv5) + keep the head trainable
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)
                 and 'resnet' in l.name), None)
    if base is None:
        raise SystemExit('could not locate the ResNet50 base sub-model')
    n_unfrozen = 0
    for l in base.layers:
        if 'conv5' in l.name:
            l.trainable = True
            n_unfrozen += 1
    print(f'    unfroze {n_unfrozen} conv5-stage layers (others stay frozen)',
          flush=True)

    model.compile(optimizer=Adam(LR), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_eurosat()
    ds_tr = make_ds(x_tr, y_tr, BATCH, shuffle=True, augment=True)
    ds_va = make_ds(x_va, y_va, BATCH)
    ds_te = make_ds(x_te, y_te, BATCH)

    print('[2/4] fine-tuning (conv5 + head, lr=1e-4, augmented)...', flush=True)
    # ModelCheckpoint saves the BEST-weights h5 every epoch, so a kill mid-run
    # (thermal throttling / session drop) never loses the best model.
    model.fit(ds_tr, validation_data=ds_va, epochs=MAX_EPOCHS, verbose=1,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      OUT, monitor='val_accuracy', save_best_only=True,
                      mode='max', verbose=1),
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_accuracy', patience=EARLY_PATIENCE,
                      mode='max', restore_best_weights=True, verbose=1),
              ])

    tr = model.evaluate(ds_tr, verbose=0)[1]
    va = model.evaluate(ds_va, verbose=0)[1]
    te = model.evaluate(ds_te, verbose=0)[1]
    print(f'[3/4] EuroSAT accuracy -- train {tr:.4f} | val {va:.4f} | '
          f'test {te:.4f}', flush=True)

    print(f'[4/4] saving -> {OUT} ...', flush=True)
    model.save(OUT)
    print(f'RESULT test_acc={te:.4f} path={OUT}', flush=True)


if __name__ == '__main__':
    main()
