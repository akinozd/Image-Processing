import cv2
import numpy as np

img = cv2.imread("yuz_input.jpg")

# 1. RGB Uzayinda Ten Tespiti
B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
maske_rgb = (
    (R > 95) & (G > 40) & (B > 20) &
    ((np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) > 15) &
    (np.abs(R.astype(int) - G.astype(int)) > 15) &
    (R > G) & (R > B)
)
sonuc_rgb = img.copy()
sonuc_rgb[~maske_rgb] = (0, 0, 255)

# 2. YCbCr Uzayinda Ten Tespiti
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
maske_ycbcr = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)
sonuc_ycbcr = img.copy()
sonuc_ycbcr[~maske_ycbcr] = (0, 0, 255)

# 3. HSV Uzayinda Ten Tespiti
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
maske_hsv = (H <= 25) & (S >= 40) & (S <= 250) & (V >= 50)
sonuc_hsv = img.copy()
sonuc_hsv[~maske_hsv] = (0, 0, 255)

# Sonuclari kaydet
cv2.imwrite("cikti_task5_rgb.jpg", sonuc_rgb)
cv2.imwrite("cikti_task5_ycbcr.jpg", sonuc_ycbcr)
cv2.imwrite("cikti_task5_hsv.jpg", sonuc_hsv)

print("Task 5 bitti.")