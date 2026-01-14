import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Configuración para correr con CPU
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(4)

# CONFIG (AJUSTADO A CPU)
IMG_SIZE = 96          # reduce costo computacional
BATCH_SIZE = 8         # crítico en CPU
EPOCHS = 30            # EarlyStopping corta antes
LR = 1e-3

DATA_PATH = "datos_entrenamiento/"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# DEGRADATION FUNCTION
# =========================
def degrade_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)

    noise = np.random.normal(0, 10, img.shape)
    img = img + noise

    h, w = img.shape
    x = np.random.randint(0, w // 4)
    y = np.random.randint(0, h // 4)

    img = img[y:y + int(0.8 * h), x:x + int(0.8 * w)]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

# =========================
# DATA LOADING
# =========================
def load_data():
    X, y = [], []

    for fname in os.listdir(DATA_PATH):
        if not fname.lower().endswith((".png", ".jpg", ".bmp")):
            continue

        path = os.path.join(DATA_PATH, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        degraded = degrade_image(img)

        X.append(degraded)
        y.append(img)

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y, dtype="float32") / 255.0

    X = X[..., np.newaxis]
    y = y[..., np.newaxis]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo U-NET LIGERO
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c2])
    c4 = conv_block(u2, 64)

    u1 = layers.UpSampling2D()(c4)
    u1 = layers.Concatenate()([u1, c1])
    c5 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return models.Model(inputs, outputs)

# Loss ANTI-BLUR
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return 0.7 * mse + 0.3 * ssim

# Entrenamiento
def main():
    X_tr, X_val, y_tr, y_val = load_data()


    model = build_unet()
    model.compile(
        optimizer=Adam(LR),
        loss=combined_loss,
        metrics=["mae"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_model_cpu.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
