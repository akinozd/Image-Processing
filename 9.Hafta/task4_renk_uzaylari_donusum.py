import cv2
import numpy as np

# 1. Resmi BMP olarak kaydet ve ac
img = cv2.imread("input.jpg")
cv2.imwrite("input_renkli.bmp", img)
bmp_img = cv2.imread("input_renkli.bmp")

# 2. HSV, YUV ve CIE Lab donusumleri
hsv = cv2.cvtColor(bmp_img, cv2.COLOR_BGR2HSV)
yuv = cv2.cvtColor(bmp_img, cv2.COLOR_BGR2YUV)
lab = cv2.cvtColor(bmp_img, cv2.COLOR_BGR2Lab)

# 3. HSI donusumu (Manuel formulle)
bgr = bmp_img.astype(np.float32) / 255.0
B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

I = (R + G + B) / 3.0
minimum = np.minimum(np.minimum(R, G), B)
S = 1.0 - (3.0 / (R + G + B + 1e-6)) * minimum

pay = 0.5 * ((R - G) + (R - B))
payda = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
theta = np.arccos(np.clip(pay / payda, -1.0, 1.0))
H = np.where(B <= G, theta, 2 * np.pi - theta)
H = H / (2 * np.pi)

hsi = (np.stack([H, S, I], axis=2) * 255).astype(np.uint8)

# 4. Hepsini BMP olarak kaydet
cv2.imwrite("cikti_task4_hsv.bmp", hsv)
cv2.imwrite("cikti_task4_yuv.bmp", yuv)
cv2.imwrite("cikti_task4_lab.bmp", lab)
cv2.imwrite("cikti_task4_hsi.bmp", hsi)

print("Task 4 bitti: HSV, YUV, Lab ve HSI BMP dosyalari kaydedildi.")