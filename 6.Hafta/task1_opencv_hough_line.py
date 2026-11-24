import cv2
import numpy as np

img = cv2.imread("input.jpg")
gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny ile kenar bulma
kenar = cv2.Canny(gri, 50, 150)

# Ince kenarin median filtrede silinmemesi icin hafif genisletip median filtre uygula
kenar = cv2.dilate(kenar, None, iterations=1)
kenar = cv2.medianBlur(kenar, 3)

# 1. Fonksiyon: Standart HoughLines (Mavi)
resim1 = img.copy()
cizgiler = cv2.HoughLines(kenar, 1, np.pi / 180, 100)
if cizgiler is not None:
    for cizgi in cizgiler:
        r, theta = cizgi.reshape(-1)
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * r, b * r
        p1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        p2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv2.line(resim1, p1, p2, (255, 0, 0), 3)

# 2. Fonksiyon: Olasiliksal HoughLinesP (Yesil)
resim2 = img.copy()
parcalar = cv2.HoughLinesP(kenar, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
if parcalar is not None:
    for parca in parcalar:
        x1, y1, x2, y2 = parca.reshape(-1)
        cv2.line(resim2, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imwrite("cikti_task1_hough1.jpg", resim1)
cv2.imwrite("cikti_task1_hough2.jpg", resim2)
print("Task 1 bitti.")