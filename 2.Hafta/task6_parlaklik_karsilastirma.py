import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

# 1. Sadece 0.75 ile carpma islemi
sadece_carpim = img * 0.75
sadece_carpim = np.clip(sadece_carpim, 0, 255).astype(np.uint8)

# 2. 0.75 ile carpip 20 ekleme islemi
carpim_ekleme = (img * 0.75) + 20
carpim_ekleme = np.clip(carpim_ekleme, 0, 255).astype(np.uint8)

# Sayisal olarak ortalama, en kucuk ve en buyuk piksel degerlerini yazdir
print("--- Sayisal Karsilastirma ---")
print(f"Orijinal Resim -> Ortalama: {np.mean(img):.2f}, Min: {np.min(img)}, Max: {np.max(img)}")
print(f"0.75 * f       -> Ortalama: {np.mean(sadece_carpim):.2f}, Min: {np.min(sadece_carpim)}, Max: {np.max(sadece_carpim)}")
print(f"0.75 * f + 20  -> Ortalama: {np.mean(carpim_ekleme):.2f}, Min: {np.min(carpim_ekleme)}, Max: {np.max(carpim_ekleme)}")

cv2.imwrite("cikti_task6_sadece_075.jpg", sadece_carpim)
cv2.imwrite("cikti_task6_075_arti_20.jpg", carpim_ekleme)
print("Task 6 tamamlandi: Gorseller kaydedildi.")