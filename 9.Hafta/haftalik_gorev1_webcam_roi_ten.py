import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, onceki_kare = cap.read()
onceki_gri = cv2.cvtColor(onceki_kare, cv2.COLOR_BGR2GRAY)
onceki_gri = cv2.GaussianBlur(onceki_gri, (21, 21), 0)

while True:
    ret, kare = cap.read()
    if not ret:
        break

    h, w = kare.shape[:2]
    # Ekranin ortasinda bir ROI (Ilgi Alani) dikdortgeni belirle
    x1, y1, x2, y2 = w // 4, h // 6, 3 * w // 4, 5 * h // 6

    # 1. Hareketli bolge tespiti (Arka plan cikarma)
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    gri = cv2.GaussianBlur(gri, (21, 21), 0)
    fark = cv2.absdiff(onceki_gri, gri)
    _, hareket_maskesi = cv2.threshold(fark, 25, 255, cv2.THRESH_BINARY)
    hareket_maskesi = cv2.dilate(hareket_maskesi, None, iterations=2)

    # Hareket etmeyen arka plani siyah yap
    sonuc = np.zeros_like(kare)
    sonuc[hareket_maskesi > 0] = kare[hareket_maskesi > 0]

    # 2. Sadece ROI icinde insan tenini tespit et (YCbCr ile) ve renklendir
    roi = kare[y1:y2, x1:x2]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Cr, Cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    ten_maskesi = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)

    roi_sonuc = sonuc[y1:y2, x1:x2]
    # Tespit edilen ten bolgesini yesil tonla renklendir
    roi_sonuc[ten_maskesi] = (0, 255, 0)
    sonuc[y1:y2, x1:x2] = roi_sonuc

    # ROI sinirini ciz
    cv2.rectangle(sonuc, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("ROI Ten ve Hareket Tespiti (Cikis: q)", sonuc)

    onceki_gri = gri
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()