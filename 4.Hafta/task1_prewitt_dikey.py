import cv2
import numpy as np

# 1. Gri tonlamalı oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 2. Tuğla yüzeyindeki pürüzleri bastırmak için Gauss çekirdeğini (5, 5) yapıyoruz
blurred = cv2.GaussianBlur(img, (5, 5), 1.5)

# 3. Hocanın verdiği 3x3 Dikey Prewitt Çekirdeği
kernel_prewitt_v = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)

# Konvolüsyon
prewitt_v = cv2.filter2D(blurred, cv2.CV_64F, kernel_prewitt_v)
prewitt_v_abs = np.abs(prewitt_v)

# 4. Dersteki slaytta olduğu gibi (99.5th percentile veya max ile normalize ederek temizleme)
norm_img = (prewitt_v_abs / (np.max(prewitt_v_abs) + 1e-5)) * 255.0
norm_img = norm_img.astype(np.uint8)

# Arka plandaki mikro-kumlanmayı kesmek için hafif alt eşik (Noise floor)
_, temiz_kenar = cv2.threshold(norm_img, 40, 255, cv2.THRESH_TOZERO)

cv2.imwrite("cikti_task1_prewitt_dikey.jpg", temiz_kenar)
print("Task 1 tamamlandı: 'cikti_task1_prewitt_dikey.jpg' temiz şekilde kaydedildi.")