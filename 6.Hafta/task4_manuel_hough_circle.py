import cv2
import numpy as np

# Sadece senin koydugun goruntuyu okur, yeni dosya olusturmaz
img = cv2.imread("cember_tekil_input.jpg")
h_orj, w_orj = img.shape[:2]

# 3D oylama dongusunun hizli calismasi icin resmi kucultuyoruz
olcek = 120 / w_orj
kucuk = cv2.resize(img, (0, 0), fx=olcek, fy=olcek)
gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)

kenar = cv2.Canny(gri, 50, 150)
h, w = kenar.shape

# 1. Sin ve Cos degerlerini bastan hesaplayip bellekte tut
acilar = np.deg2rad(np.arange(0, 360, 5))
cos_tablo = np.cos(acilar)
sin_tablo = np.sin(acilar)

# 2. 3D Oylama matrisi olustur
min_r = 10
max_r = min(h, w) // 2
yaricaplar = np.arange(min_r, max_r, 1)
oy_matrisi = np.zeros((h, w, len(yaricaplar)), dtype=np.int32)

y_ler, x_ler = np.nonzero(kenar)

for i in range(len(x_ler)):
    x = x_ler[i]
    y = y_ler[i]
    for r_idx in range(len(yaricaplar)):
        r = yaricaplar[r_idx]
        for t in range(len(acilar)):
            a = int(round(x - r * cos_tablo[t]))
            b = int(round(y - r * sin_tablo[t]))
            if 0 <= a < w and 0 <= b < h:
                oy_matrisi[b, a, r_idx] += 1

# 3. En yuksek oyu alan cemberi bul ve orijinal resme ciz
sonuc = img.copy()
esik = int(oy_matrisi.max() * 0.90)

for r_idx in range(len(yaricaplar)):
    katman = oy_matrisi[:, :, r_idx]
    if np.max(katman) >= esik:
        b, a = np.unravel_index(np.argmax(katman), katman.shape)
        merkez_x = int(a / olcek)
        merkez_y = int(b / olcek)
        yaricap = int(yaricaplar[r_idx] / olcek)
        cv2.circle(sonuc, (merkez_x, merkez_y), yaricap, (255, 0, 0), 3)
        cv2.circle(sonuc, (merkez_x, merkez_y), 4, (0, 0, 255), -1)

cv2.imwrite("cikti_task4_manuel_cember.jpg", sonuc)
print("Task 4 bitti.")