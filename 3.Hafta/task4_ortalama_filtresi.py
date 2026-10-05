import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 1. Test için yapay gürültü ekleme
np.random.seed(42)
noise = np.random.normal(0, 25, img.shape)
gurultulu_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
cv2.imwrite("cikti_task4_gurultulu_resim.jpg", gurultulu_img)

# 2. 3x3 Ortalama Filtresi (Manuel Konvolüsyon)
h, w = gurultulu_img.shape
filtered_img = np.zeros((h, w), dtype=np.uint8)
padded_img = np.pad(gurultulu_img, pad_width=1, mode='edge')

for r in range(h):
    for c in range(w):
        bolge = padded_img[r : r + 3, c : c + 3]
        filtered_img[r, c] = int(np.sum(bolge) / 9.0)

cv2.imwrite("cikti_task4_filtrelenmis_resim.jpg", filtered_img)
print("Task 4 tamamlandı: 'cikti_task4_filtrelenmis_resim.jpg' kaydedildi.")