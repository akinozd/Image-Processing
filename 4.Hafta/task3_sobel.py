import cv2
import numpy as np

# 1. Gri tonlamalı görseli oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 2. Ön yumuşatma (Gauss Filtresi)
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

# 3. 3x3 Sobel Çekirdekleri
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

sobel_y_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)

# Konvolüsyon işlemleri
gx = cv2.filter2D(blurred, cv2.CV_64F, sobel_x_kernel)
gy = cv2.filter2D(blurred, cv2.CV_64F, sobel_y_kernel)

gx_abs = np.abs(gx)
gy_abs = np.abs(gy)

# Bileşke gradyan büyüklüğü (Magnitude)
magnitude = np.sqrt(gx**2 + gy**2)

# 4. Normalize etme ve eşikleme fonksiyonu
def temizle_kenar(gorsel, th=40):
    norm = (gorsel / (np.max(gorsel) + 1e-5)) * 255.0
    norm = norm.astype(np.uint8)
    _, res = cv2.threshold(norm, th, 255, cv2.THRESH_TOZERO)
    return res

sobel_x_temiz = temizle_kenar(gx_abs)
sobel_y_temiz = temizle_kenar(gy_abs)
sobel_mag_temiz = temizle_kenar(magnitude)

# Çıktıları kaydet
cv2.imwrite("cikti_task3_sobel_x.jpg", sobel_x_temiz)
cv2.imwrite("cikti_task3_sobel_y.jpg", sobel_y_temiz)
cv2.imwrite("cikti_task3_sobel_bileske.jpg", sobel_mag_temiz)

print("Task 3 tamamlandı: Sobel X, Y ve Bileşke kenar görselleri kaydedildi.")