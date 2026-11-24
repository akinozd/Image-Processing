import cv2
import numpy as np
import threading
import time

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

yukseklik, genislik = img.shape
parca_boyutu = yukseklik // 4

# --- 1. TEK THREAD (SIRALI) HISTOGRAM HESAPLAMA ---
t1_basla = time.time()

hist_tek = np.zeros(256, dtype=np.int64)
for i in range(yukseklik):
    for j in range(genislik):
        deger = img[i, j]
        hist_tek[deger] += 1

sure_1_thread = (time.time() - t1_basla) * 1000

# --- 2. COKLU THREAD (4 THREAD) HISTOGRAM HESAPLAMA ---
yerel_histler = []
for i in range(4):
    yerel_histler.append(np.zeros(256, dtype=np.int64))

def parca_hist(resim, bas_satir, bit_satir, indeks):
    h = np.zeros(256, dtype=np.int64)
    for i in range(bas_satir, bit_satir):
        for j in range(genislik):
            deger = resim[i, j]
            h[deger] += 1
    yerel_histler[indeks] = h

threadler = []
t4_basla = time.time()

for i in range(4):
    bas = i * parca_boyutu
    son = yukseklik if i == 3 else (i + 1) * parca_boyutu
    t = threading.Thread(target=parca_hist, args=(img, bas, son, i))
    threadler.append(t)
    t.start()

for t in threadler:
    t.join()

# 4 parcayi birlestir
hist_coklu = np.zeros(256, dtype=np.int64)
for i in range(4):
    hist_coklu += yerel_histler[i]

sure_4_thread = (time.time() - t4_basla) * 1000

print(f"Histogram 1 Thread Calisma Suresi: {sure_1_thread:.2f} ms")
print(f"Histogram 4 Thread Calisma Suresi: {sure_4_thread:.2f} ms")