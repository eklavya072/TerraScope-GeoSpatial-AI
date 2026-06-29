import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from config import MODEL_CONFIG, CLASS_NAMES

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

def load_eurosat_in_memory():
    print("Loading EuroSAT RGB from Hugging Face into memory...")
    from datasets import load_dataset
    hf = load_dataset("giswqs/EuroSAT_RGB")

    def _load(split):
        ds = hf[split]
        images, labels = [], []
        for i in range(len(ds)):
            img = np.array(ds[i]['image'], dtype=np.float32) / 255.0
            images.append(img)
            labels.append(ds[i]['label'])
        return np.array(images), np.array(labels)

    x_train, y_train = _load('train')
    x_val, y_val = _load('validation')
    x_test, y_test = _load('test')

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def make_ds(x, y, batch_size, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(5000)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda img, lbl: (aug(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_simple_cnn(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def train():
    print("Step 1: Loading EuroSAT into memory...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_eurosat_in_memory()

    ds_train = make_ds(x_train, y_train, BATCH_SIZE, shuffle=True, augment=True)
    ds_val = make_ds(x_val, y_val, BATCH_SIZE)
    ds_test = make_ds(x_test, y_test, BATCH_SIZE)

    print("Step 2: Building lightweight CNN...")
    model = build_simple_cnn(MODEL_CONFIG['input_shape'], 10)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    print("Step 3: Training...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_CONFIG['best_model_path'],
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(ds_train, validation_data=ds_val,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)

    print("Step 4: Evaluating on test set...")
    test_loss, test_acc = model.evaluate(ds_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    print("Step 5: Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_CONFIG['model_path'])
    print(f"Model saved to {MODEL_CONFIG['model_path']}")

    np.save(MODEL_CONFIG['indices_path'], CLASS_NAMES)
    print(f"Class indices saved to {MODEL_CONFIG['indices_path']}")
    print("Training complete!")

if __name__ == "__main__":
    train()
