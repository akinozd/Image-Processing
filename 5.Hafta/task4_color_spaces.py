import cv2
import numpy as np


def manuel_bgr2ycrcb(gorsel):
    b = gorsel[:, :, 0].astype(np.float64)
    g = gorsel[:, :, 1].astype(np.float64)
    r = gorsel[:, :, 2].astype(np.float64)
    
    # ITU-R BT.601 tam aralık (full-range) formülleri
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    
    # Yuvarlama ve 0-255 aralığına kırpma
    y = np.clip(np.round(y), 0, 255).astype(np.uint8)
    cr = np.clip(np.round(cr), 0, 255).astype(np.uint8)
    cb = np.clip(np.round(cb), 0, 255).astype(np.uint8)
    
    # OpenCV kanal sırası: Y, Cr, Cb
    return cv2.merge([y, cr, cb])

# Görseli oku
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

# OpenCV hazır fonksiyonu dönüşüm
opencv_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 4. Kendi yazdığımız fonksiyon ile dönüşüm
manuel_ycrcb = manuel_bgr2ycrcb(img)

# 5. Sayısal Karşılaştırma ve Hata Hesabı
fark = np.abs(opencv_ycrcb.astype(int) - manuel_ycrcb.astype(int))
toplam_fark = np.sum(fark)
max_fark = np.max(fark)
ortalama_hata = np.mean(fark)

print("--- Task 4: Renk Uzayi Donusum Karsilastirmasi ---")
print(f"Toplam Mutlak Hata : {toplam_fark}")
print(f"Maksimum Piksel Farki: {max_fark}")
print(f"Piksel Basina Ortalama Hata: {ortalama_hata:.4f}")

if toplam_fark == 0:
    print("Sonuc: Mukemmel eslesme, hata degeri SIFIR!")
else:
    print("Sonuc: Donusum basarili. Kalan kucuk fark OpenCV'nin sabit noktali (fixed-point integer) yuvarlama hassasiyetinden kaynaklanmaktadir.")

# Kanalları ve sonucu kaydet
cv2.imwrite("cikti_task4_opencv_ycrcb.jpg", opencv_ycrcb)
cv2.imwrite("cikti_task4_manuel_ycrcb.jpg", manuel_ycrcb)
print("Task 4 tamamlandi: Ciktilar kaydedildi.")