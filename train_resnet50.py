"""Retrain the TerraScope classifier on a ResNet50 backbone.

Why this exists: the served model (models/mobilenetv2_eurosat.h5) is the
lightweight MobileNetV2 (2.3M params). The Evros CNN experiments used it, so
the 'CNN ceiling' we measured is the CHEAPEST model's ceiling. This script
retrains the same EuroSAT 10-class task on the much stronger ResNet50 (25M
params) and saves a drop-in replacement model.

CPU tractability: full 30-epoch fine-tuning of ResNet50 on CPU would take
many hours, so this uses the standard transfer-learning recipe that is
methodologically equivalent for the frozen-backbone phase:

  1. Load giswqs/EuroSAT_RGB (train/val/test), normalized /255 like the
     original training.
  2. FREEZE the ImageNet-pretrained ResNet50 base and extract GAP features
     once for all splits (cached to models/resnet50_features.npz so re-runs
     skip the expensive pass).
  3. Train the classification head (Dense 256 -> Dropout 0.3 -> Dense 10)
     on the features with early stopping.
  4. Assemble the full functional model (input -> base -> GAP -> head),
     reuse the trained head weights, and save models/resnet50_eurosat.h5.

No augmentation (CPU budget); Dropout + early stopping provide
regularization. The head has the same architecture as the original
MobileNetV2 build for an apples-to-apples comparison.

Usage:
    python train_resnet50.py            # extract features (if needed) + train
    python train_resnet50.py --no-extract   # reuse cached features
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from config import CLASS_NAMES
from model_handler import resnet50_preprocess

# v2: features are extracted from PREPROCESSED inputs (ImageNet caffe mode:
# x*255 -> RGB->BGR -> mean subtraction). Frozen ImageNet BN stats are only
# meaningful in that space -- v1 extracted on raw [0,1] images and the head
# capped at 60% test accuracy.
FEATURES_CACHE = os.path.join('models', 'resnet50_features_v2.npz')
OUT_PATH = os.path.join('models', 'resnet50_eurosat.h5')
BATCH = 128
LR = 1e-3
MAX_EPOCHS = 40


def load_eurosat():
    print('[1/4] loading giswqs/EuroSAT_RGB (cached) ...', flush=True)
    from datasets import load_dataset
    hf = load_dataset('giswqs/EuroSAT_RGB')

    def _load(split):
        ds = hf[split]
        imgs = np.array([np.array(ds[i]['image'], dtype=np.float32) / 255.0
                         for i in range(len(ds))])
        return imgs, np.array([ds[i]['label'] for i in range(len(ds))])

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (
        _load(s) for s in ('train', 'validation', 'test'))
    print(f'    train {x_tr.shape} | val {x_va.shape} | test {x_te.shape}',
          flush=True)
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def get_features(images, fe):
    """GAP-pooled ResNet50 features in batches (CPU)."""
    out = []
    for i in range(0, len(images), BATCH):
        out.append(fe.predict(images[i:i + BATCH], verbose=0))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-extract', action='store_true',
                    help='reuse cached features instead of extracting')
    args = ap.parse_args()

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_eurosat()

    # ---------- frozen backbone + features ----------
    print('[2/4] building frozen ResNet50 (ImageNet weights) ...', flush=True)
    # The model consumes RAW [0,1] RGB (same convention as the served
    # MobileNetV2 and the pipeline), and does the ImageNet preprocessing
    # (x*255 -> RGB->BGR -> mean subtraction) INSIDE the graph so frozen
    # ImageNet BN statistics see their calibrated input space. The function
    # is registered as serializable in model_handler so the saved h5 loads.
    inputs = tf.keras.Input((64, 64, 3))
    x = layers.Lambda(resnet50_preprocess)(inputs)
    base = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    base.trainable = False
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    fe = tf.keras.Model(inputs, x)

    if args.no_extract and os.path.exists(FEATURES_CACHE):
        print('    reusing cached features', flush=True)
        z = np.load(FEATURES_CACHE)
        f_tr, f_va, f_te = z['f_tr'], z['f_va'], z['f_te']
    elif os.path.exists(FEATURES_CACHE):
        print(f'    cached features exist ({FEATURES_CACHE}) -- loading',
              flush=True)
        z = np.load(FEATURES_CACHE)
        f_tr, f_va, f_te = z['f_tr'], z['f_va'], z['f_te']
    else:
        t0 = time.time()
        print(f'    extracting features for {len(x_tr) + len(x_va) + len(x_te)} '
              f'images on CPU (one-time, ~15-40 min) ...', flush=True)
        f_tr = get_features(x_tr, fe)
        f_va = get_features(x_va, fe)
        f_te = get_features(x_te, fe)
        print(f'    extraction done in {(time.time() - t0) / 60:.1f} min',
              flush=True)
        np.savez(FEATURES_CACHE, f_tr=f_tr, f_va=f_va, f_te=f_te,
                 y_tr=y_tr, y_va=y_va, y_te=y_te)

    # ---------- head training on features ----------
    print('[3/4] training classification head on features ...', flush=True)
    head = tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASS_NAMES), activation='softmax'),
    ])
    head.compile(optimizer=Adam(LR), loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    head.fit(f_tr, y_tr, validation_data=(f_va, y_va), epochs=MAX_EPOCHS,
             batch_size=BATCH, verbose=1,
             callbacks=[tf.keras.callbacks.EarlyStopping(
                 monitor='val_accuracy', patience=6, mode='max',
                 restore_best_weights=True, verbose=1)])

    tr_acc = head.evaluate(f_tr, y_tr, verbose=0)[1]
    va_acc = head.evaluate(f_va, y_va, verbose=0)[1]
    te_acc = head.evaluate(f_te, y_te, verbose=0)[1]
    print(f'    EuroSAT accuracy -- train {tr_acc:.4f} | '
          f'val {va_acc:.4f} | test {te_acc:.4f}', flush=True)

    # ---------- assemble + save full model ----------
    print(f'[4/4] assembling full model -> {OUT_PATH} ...', flush=True)
    inputs = tf.keras.Input((64, 64, 3))
    x = layers.Lambda(resnet50_preprocess)(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    # reuse the SAME head layers (weights already trained above)
    for layer in head.layers:
        x = layer(x)
    full = tf.keras.Model(inputs, x)
    os.makedirs('models', exist_ok=True)
    full.save(OUT_PATH)
    print('saved.', flush=True)
    print(f'RESULT test_acc={te_acc:.4f} path={OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
