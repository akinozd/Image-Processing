import cv2
import numpy as np

# 1. Görüntüyü tek kanallı oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

h, w = img.shape

# 2. Min ve Max değerleri döngüyle bul
min_val = int(img[0, 0])
max_val = int(img[0, 0])

for r in range(h):
    for c in range(w):
        val = int(img[r, c])
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

print(f"Task 1 -> Min: {min_val}, Max: {max_val}")

# 3. [0, 255] aralığına doğrusal ölçekleme (hazır fonksiyonsuz)
if max_val != min_val:
    scaled_img = ((img.astype(np.float32) - min_val) / (max_val - min_val)) * 255.0
    scaled_img = np.round(scaled_img).astype(np.uint8)
else:
    scaled_img = img.copy()

cv2.imwrite("cikti_task1_dogrusal_olcekleme.jpg", scaled_img)
print("Task 1 tamamlandı: 'cikti_task1_dogrusal_olcekleme.jpg' kaydedildi.")