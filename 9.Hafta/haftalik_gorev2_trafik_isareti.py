import cv2
import numpy as np

img = cv2.imread("trafik_input.jpg")
sonuc = img.copy()

# 1. HSV renk uzayinda kirmizi rengi maskele
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
maske_hsv = cv2.inRange(hsv, np.array([0, 90, 70]), np.array([10, 255, 255])) | \
            cv2.inRange(hsv, np.array([170, 90, 70]), np.array([180, 255, 255]))

# 2. YCrCb renk uzayinda Cr kanali ile destekle
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
_, Cr, _ = cv2.split(ycrcb)
_, maske_cr = cv2.threshold(Cr, 150, 255, cv2.THRESH_BINARY)

kirmizi_maske = cv2.bitwise_and(maske_hsv, maske_cr)
kirmizi_maske = cv2.GaussianBlur(kirmizi_maske, (9, 9), 2)

# 3. Hough Circle ile kirmizi cemberi tespit et
cemberler = cv2.HoughCircles(
    kirmizi_maske,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=100,
    param2=25,
    minRadius=15,
    maxRadius=200
)

if cemberler is not None:
    cemberler = np.uint16(np.around(cemberler))
    for x, y, r in cemberler[0, :]:
        cv2.circle(sonuc, (x, y), r, (0, 255, 0), 3)
        cv2.circle(sonuc, (x, y), 3, (255, 0, 0), -1)

cv2.imwrite("cikti_haftalik2_tespit.jpg", sonuc)
print("Haftalik Gorev 2 bitti.")