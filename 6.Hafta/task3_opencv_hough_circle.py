import cv2
import numpy as np

img = cv2.imread("cember_input.jpg")
gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Canny ile kenar bulma ve median filtre
kenar = cv2.Canny(gri, 50, 150)
kenar = cv2.medianBlur(kenar, 3)

# 2. OpenCV HoughCircles ile cember tespiti (esik degerini yukseltip gurultuyu eledik)
cemberler = cv2.HoughCircles(kenar, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=70, minRadius=20, maxRadius=450)

sonuc = img.copy()
if cemberler is not None:
    cemberler = np.uint16(np.around(cemberler))
    for x, y, r in cemberler[0, :]:
        cv2.circle(sonuc, (x, y), r, (0, 255, 0), 4)
        cv2.circle(sonuc, (x, y), 4, (0, 0, 255), -1)

cv2.imwrite("cikti_task3_opencv_cember.jpg", sonuc)
print("Task 3 bitti.")