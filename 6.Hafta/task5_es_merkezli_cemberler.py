import cv2
import numpy as np

# Klasordeki es merkezli halkalardan olusan resmi okuyoruz
img = cv2.imread("cember_input.jpg")
h_orj, w_orj = img.shape[:2]

olcek = 120 / w_orj
kucuk = cv2.resize(img, (0, 0), fx=olcek, fy=olcek)
gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)

kenar = cv2.Canny(gri, 50, 150)
h, w = kenar.shape

# 1. Sin ve Cos tablosunu bellekte tut
acilar = np.deg2rad(np.arange(0, 360, 5))
cos_tablo = np.cos(acilar)
sin_tablo = np.sin(acilar)

# 2. 3D Oylama matrisi olustur
min_r = 8
max_r = min(h, w) // 2
yaricaplar = np.arange(min_r, max_r, 3)
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

# 3. Tum yaricap katmanlarinin toplam oyundan ortak ana merkezi bul
toplam_merkez_oyu = np.sum(oy_matrisi, axis=2)
merkez_b, merkez_a = np.unravel_index(np.argmax(toplam_merkez_oyu), toplam_merkez_oyu.shape)

sonuc = img.copy()
merkez_x = int(merkez_a / olcek)
merkez_y = int(merkez_b / olcek)

# Ortak merkezdeki (ve 2 piksel komsulugundaki) yuksek oylu es merkezli yaricaplari sec
bolge = oy_matrisi[max(0, merkez_b-2):merkez_b+3, max(0, merkez_a-2):merkez_a+3, :]
yaricap_oylari = np.max(bolge, axis=(0, 1))
esik = int(np.max(yaricap_oylari) * 0.80)

for r_idx in range(len(yaricaplar)):
    if yaricap_oylari[r_idx] >= esik:
        yaricap = int(yaricaplar[r_idx] / olcek)
        cv2.circle(sonuc, (merkez_x, merkez_y), yaricap, (0, 255, 0), 3)

cv2.circle(sonuc, (merkez_x, merkez_y), 5, (0, 0, 255), -1)
cv2.imwrite("cikti_task5_es_merkezli.jpg", sonuc)
print("Task 5 bitti.")