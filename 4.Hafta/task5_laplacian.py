import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# İkinci türev gürültüye çok duyarlıdır, bu yüzden Gauss ön filtreleme şarttır
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

# Standart 3x3 Laplacian Çekirdeği
laplacian_kernel = np.array([[ 0,  1,  0],
                             [ 1, -4,  1],
                             [ 0,  1,  0]], dtype=np.float32)

laplacian = cv2.filter2D(blurred, cv2.CV_64F, laplacian_kernel)
laplacian_abs = np.abs(laplacian)

norm = (laplacian_abs / (np.max(laplacian_abs) + 1e-5)) * 255.0
_, laplacian_temiz = cv2.threshold(norm.astype(np.uint8), 35, 255, cv2.THRESH_TOZERO)

cv2.imwrite("cikti_task5_laplacian.jpg", laplacian_temiz)
print("Task 5 tamamlandı: 'cikti_task5_laplacian.jpg' kaydedildi.")