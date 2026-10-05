import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

h, w = img.shape
total_pixels = h * w

# 1. 256 elemanlı histogramı manuel çıkar
hist = [0] * 256
for r in range(h):
    for c in range(w):
        hist[img[r, c]] += 1

# 2. Kümülatif Dağılım Fonksiyonu (CDF) hesapla
cdf = [0] * 256
cumulative_sum = 0
for i in range(256):
    cumulative_sum += hist[i]
    cdf[i] = cumulative_sum

cdf_min = next(v for v in cdf if v > 0)

# 3. Eşitleme haritalama tablosu oluştur
lookup_table = np.zeros(256, dtype=np.uint8)
if total_pixels != cdf_min:
    for v in range(256):
        mapped = round(((cdf[v] - cdf_min) / (total_pixels - cdf_min)) * 255)
        lookup_table[v] = np.clip(mapped, 0, 255)
else:
    lookup_table = np.arange(256, dtype=np.uint8)

# Haritayı uygula
equalized_img = lookup_table[img]
cv2.imwrite("cikti_task2_histogram_esitleme.jpg", equalized_img)
print("Task 2 tamamlandı: 'cikti_task2_histogram_esitleme.jpg' kaydedildi.")