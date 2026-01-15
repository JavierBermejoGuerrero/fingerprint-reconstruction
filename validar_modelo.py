import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 96
MODEL_PATH = "outputs/best_model_denoising.h5"
VAL_PATH = "Datos-Validacion/"
OUTPUT_VAL_DIR = "outputs/validacion_resultados"

os.makedirs(OUTPUT_VAL_DIR, exist_ok=True)

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img[np.newaxis, ..., np.newaxis]

# Cargar modelo
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

ssim_scores = []
psnr_scores = []

for fname in os.listdir(VAL_PATH):
    if not fname.lower().endswith((".jpg", ".png", ".bmp")):
        continue

    path = os.path.join(VAL_PATH, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    inp = preprocess(img)
    pred = model.predict(inp, verbose=0)[0, :, :, 0]

    # Ground truth
    gt = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

    # Métricas
    ssim = tf.image.ssim(
        gt[np.newaxis, ..., np.newaxis],
        pred[np.newaxis, ..., np.newaxis],
        max_val=1.0
    ).numpy()[0]

    psnr = tf.image.psnr(
        gt[np.newaxis, ..., np.newaxis],
        pred[np.newaxis, ..., np.newaxis],
        max_val=1.0
    ).numpy()[0]

    ssim_scores.append(ssim)
    psnr_scores.append(psnr)

    print(f"{fname} → SSIM: {ssim:.4f} | PSNR: {psnr:.2f} dB")

    # =========================
    # GUARDAR IMAGEN RECONSTRUIDA
    # =========================
    pred_img = (pred * 255).clip(0, 255).astype("uint8")
    out_path = os.path.join(
        OUTPUT_VAL_DIR,
        fname.replace(".", "_reconstruida.")
    )
    cv2.imwrite(out_path, pred_img)

    # ====================
    # GUARDAR COMPARATIVA
    # ====================
    diff = np.abs(pred - gt)

    fig = plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Reconstruida")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Diferencia")
    plt.imshow(diff, cmap="hot")
    plt.axis("off")

    comp_path = os.path.join(
        OUTPUT_VAL_DIR,
        fname.replace(".", "_comparativa.")
    )
    plt.savefig(comp_path, bbox_inches="tight", dpi=150)
    plt.close()

# =========================
# PROMEDIOS FINALES
# =========================
print("\n===== RESULTADOS PROMEDIO =====")
print(f"SSIM promedio: {np.mean(ssim_scores):.4f}")
print(f"PSNR promedio: {np.mean(psnr_scores):.2f} dB")
