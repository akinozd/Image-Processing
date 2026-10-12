import cv2
import time
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 1. Canny Süre Ölçümü
t0 = time.perf_counter()
canny_edges = cv2.Canny(img, threshold1=80, threshold2=180)
t_canny = (time.perf_counter() - t0) * 1000

# 2. Sobel Süre Ölçümü (Kıyaslama)
t1 = time.perf_counter()
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
_ = np.sqrt(gx**2 + gy**2)
t_sobel = (time.perf_counter() - t1) * 1000

cv2.imwrite("cikti_task7_canny.jpg", canny_edges)
print("Task 7 tamamlandı: 'cikti_task7_canny.jpg' kaydedildi.")
print(f"-> Canny Calisma Suresi: {t_canny:.3f} ms")
print(f"-> Sobel Calisma Suresi: {t_sobel:.3f} ms")