import cv2
import numpy as np

# 1. Gri tonlamalı oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 2. Pürüzleri bastırmak için Gauss yumuşatma
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

# 3. 3x3 Yatay Prewitt Çekirdeği: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
kernel_prewitt_h = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]], dtype=np.float32)

# Konvolüsyon
prewitt_h = cv2.filter2D(blurred, cv2.CV_64F, kernel_prewitt_h)
prewitt_h_abs = np.abs(prewitt_h)

# 4. Normalize etme ve alt gürültüyü kesme
norm_img = (prewitt_h_abs / (np.max(prewitt_h_abs) + 1e-5)) * 255.0
norm_img = norm_img.astype(np.uint8)

_, temiz_kenar = cv2.threshold(norm_img, 40, 255, cv2.THRESH_TOZERO)

cv2.imwrite("cikti_task2_prewitt_yatay.jpg", temiz_kenar)
print("Task 2 tamamlandı: 'cikti_task2_prewitt_yatay.jpg' temiz şekilde kaydedildi.")