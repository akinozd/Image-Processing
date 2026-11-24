import cv2
import numpy as np

# Görüntüyü tek kanallı (gri) olarak oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

# 8-bit (0-255 -> 256 seviye) goruntuyu 6-bit (0-63 -> 64 seviye) yapma
# Her 4 degeri 1 seviyeye dusuruyoruz
img_6bit = img // 4

# Ekranda simsiyah gorunmemesi icin tekrar 0-255 araligina genisletiyoruz
cikti = (img_6bit * 4).astype(np.uint8)

cv2.imwrite("cikti_task1_6bit.jpg", cikti)
print("Task 1 tamamlandi: 'cikti_task1_6bit.jpg' kaydedildi.")