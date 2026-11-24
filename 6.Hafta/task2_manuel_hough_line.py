import cv2
import numpy as np

img = cv2.imread("input.jpg")
gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kenarlari cikar
kenar = cv2.Canny(gri, 50, 150)

h, w = kenar.shape
diag = int(np.sqrt(h**2 + w**2))

# 1. Sin ve Cos degerlerini bastan hesaplayip bellekte tut (0-180 derece)
acilar = np.deg2rad(np.arange(0, 180))
cos_tablosu = np.cos(acilar)
sin_tablosu = np.sin(acilar)

# 2. Oylama matrisini (Accumulator) olustur
# rho degeri -diag ile +diag arasinda degisir, toplam 2*diag boyutunda
oy_matrisi = np.zeros((2 * diag, 180), dtype=np.int32)

# Kenar piksellerinin (y, x) koordinatlarini bul
y_indeksler, x_indeksler = np.nonzero(kenar)

# 3. Oylama islemi: Her kenar pikseli icin tum acilarda rho hesapla ve oy ver
for i in range(len(x_indeksler)):
    x = x_indeksler[i]
    y = y_indeksler[i]
    for t in range(180):
        rho = int(round(x * cos_tablosu[t] + y * sin_tablosu[t])) + diag
        oy_matrisi[rho, t] += 1

# 4. Oylama matrisini 0-255 araligina cekip resim olarak kaydet
oy_gorsel = (oy_matrisi / oy_matrisi.max() * 255).astype(np.uint8)
cv2.imwrite("cikti_task2_oylama_matrisi.jpg", oy_gorsel)

# 5. Esik degerin uzerindeki dogrulari orijinal resim uzerine ciz
sonuc = img.copy()
esik = 180

for r in range(oy_matrisi.shape[0]):
    for t in range(180):
        if oy_matrisi[r, t] > esik:
            rho = r - diag
            a = cos_tablosu[t]
            b = sin_tablosu[t]
            x0 = a * rho
            y0 = b * rho
            p1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
            p2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
            cv2.line(sonuc, p1, p2, (0, 0, 255), 2)

cv2.imwrite("cikti_task2_manuel_dogrular.jpg", sonuc)
print("Task 2 bitti.")