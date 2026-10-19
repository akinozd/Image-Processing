import cv2
import numpy as np

# 1. Kendi Gri Tonlama Fonksiyonumuz (BT.601 Standardı)
def manuel_bgr2gray(gorsel):
    # float olarak alıp katsayılarla çarpıyoruz
    b = gorsel[:, :, 0].astype(np.float32)
    g = gorsel[:, :, 1].astype(np.float32)
    r = gorsel[:, :, 2].astype(np.float32)
    
    #BT.601 formülü: Y = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Değerleri 0-255 aralığına kırp ve uint8 yap
    return np.round(gray).astype(np.uint8)

#Renkli görseli oku
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

# opencv hazır fonksiyonu ile gri tonlama
opencv_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kendi yazdığımız fonksiyon ile gri tonlama
manuel_gray = manuel_bgr2gray(img)

# Hata Değeri
mutlak_fark_toplami = np.sum(np.abs(opencv_gray.astype(int) - manuel_gray.astype(int)))
ortalama_piksel_hatasi = mutlak_fark_toplami / img[:, :, 0].size

print(f"Toplam Mutlak Fark (Hata Degeri): {mutlak_fark_toplami}")
print(f"Piksel Basina Ortalama Hata: {ortalama_piksel_hatasi:.4f}")

# Çıktıları kaydet
cv2.imwrite("cikti_task3_opencv_gray.jpg", opencv_gray)
cv2.imwrite("cikti_task3_manuel_gray.jpg", manuel_gray)
print("Task 3 tamamlandi: Gri gorseller kaydedildi.")