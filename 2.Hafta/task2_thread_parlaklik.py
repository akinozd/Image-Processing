import cv2
import numpy as np
import threading

# Resmi gri seviye olarak oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

yukseklik, genislik = img.shape
parca_boyutu = yukseklik // 4

# 4 yatay parca icin farkli parlaklik degisimleri
parlaklik_farklari = [-50, -15, 25, 60]

def parlaklik_degistir(kaynak, bas_satir, bit_satir, fark, hedef):
    # Toplama isleminde 255'i asip tasmamasi icin int16'ya ceviriyoruz
    serit = kaynak[bas_satir:bit_satir, :].astype(np.int16) + fark
    # 0-255 araliginda sinirlayip tekrar uint8 yapiyoruz
    hedef[bas_satir:bit_satir, :] = np.clip(serit, 0, 255).astype(np.uint8)

cikti = np.zeros_like(img)
threadler = []

# 4 ayri thread olustur ve baslat
for i in range(4):
    bas = i * parca_boyutu
    son = yukseklik if i == 3 else (i + 1) * parca_boyutu
    t = threading.Thread(target=parlaklik_degistir, args=(img, bas, son, parlaklik_farklari[i], cikti))
    threadler.append(t)
    t.start()

# Tum threadlerin isini bitirmesini bekle
for t in threadler:
    t.join()

cv2.imwrite("cikti_task2_thread_parlaklik.jpg", cikti)
print("Task 2 tamamlandi: 'cikti_task2_thread_parlaklik.jpg' kaydedildi.")