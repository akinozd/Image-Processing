import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 7x7 LoG Çekirdeği Fonksiyonu (sigma = 1.4)
def olustur_log_kernel(size=7, sigma=1.4):
    k = size // 2
    y, x = np.ogrid[-k:k+1, -k:k+1]
    norm = -(1.0 / (np.pi * (sigma**4)))
    kernel = norm * (1.0 - ((x**2 + y**2) / (2.0 * (sigma**2)))) * np.exp(-(x**2 + y**2) / (2.0 * (sigma**2)))
    kernel -= kernel.mean()  # Toplamı sıfıra normalize et
    return kernel.astype(np.float32)

log_kernel = olustur_log_kernel(size=7, sigma=1.4)

# LoG filtresi doğrudan görüntüye uygulanır (Gauss kendi içinde gömülüdür)
log_result = cv2.filter2D(img, cv2.CV_64F, log_kernel)
log_abs = np.abs(log_result)

norm = (log_abs / (np.max(log_abs) + 1e-5)) * 255.0
_, log_temiz = cv2.threshold(norm.astype(np.uint8), 30, 255, cv2.THRESH_TOZERO)

cv2.imwrite("cikti_task6_log_7x7.jpg", log_temiz)
print("Task 6 tamamlandı: 'cikti_task6_log_7x7.jpg' kaydedildi.")