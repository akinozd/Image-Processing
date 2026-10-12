import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

# 1. Prewitt Bileşke
kp_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
kp_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
px = cv2.filter2D(blurred, cv2.CV_64F, kp_x)
py = cv2.filter2D(blurred, cv2.CV_64F, kp_y)
mag_prewitt = np.sqrt(px**2 + py**2)
prewitt_norm = (mag_prewitt / (np.max(mag_prewitt) + 1e-5)) * 255.0
_, prewitt_temiz = cv2.threshold(prewitt_norm.astype(np.uint8), 40, 255, cv2.THRESH_TOZERO)

# 2. Sobel Bileşke
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
sx = cv2.filter2D(blurred, cv2.CV_64F, sobel_x)
sy = cv2.filter2D(blurred, cv2.CV_64F, sobel_y)
mag_sobel = np.sqrt(sx**2 + sy**2)
sobel_norm = (mag_sobel / (np.max(mag_sobel) + 1e-5)) * 255.0
_, sobel_temiz = cv2.threshold(sobel_norm.astype(np.uint8), 40, 255, cv2.THRESH_TOZERO)

# Yan yana birleştir (Sol: Prewitt, Sağ: Sobel)
karsilastirma = np.hstack((prewitt_temiz, sobel_temiz))
cv2.imwrite("cikti_task4_prewitt_vs_sobel.jpg", karsilastirma)
print("Task 4 tamamlandı: 'cikti_task4_prewitt_vs_sobel.jpg' kaydedildi.")