import cv2
import numpy as np
import threading
import time

# Resmi gri seviye olarak oku
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

yukseklik, genislik = img.shape
parca_boyutu = yukseklik // 4
parlaklik_farklari = [-50, -15, 25, 60]

def parlaklik_degistir(kaynak, bas_satir, bit_satir, fark, hedef):
    serit = kaynak[bas_satir:bit_satir, :].astype(np.int16) + fark
    hedef[bas_satir:bit_satir, :] = np.clip(serit, 0, 255).astype(np.uint8)

# --- 1. TEK THREAD (SIRALI) OLCUMU ---
cikti_tek = np.zeros_like(img)
t1_basla = time.time()

for i in range(4):
    bas = i * parca_boyutu
    son = yukseklik if i == 3 else (i + 1) * parca_boyutu
    parlaklik_degistir(img, bas, son, parlaklik_farklari[i], cikti_tek)

sure_1_thread = (time.time() - t1_basla) * 1000

# --- 2. COKLU THREAD (4 THREAD) OLCUMU ---
cikti_coklu = np.zeros_like(img)
threadler = []
t4_basla = time.time()

for i in range(4):
    bas = i * parca_boyutu
    son = yukseklik if i == 3 else (i + 1) * parca_boyutu
    t = threading.Thread(target=parlaklik_degistir, args=(img, bas, son, parlaklik_farklari[i], cikti_coklu))
    threadler.append(t)
    t.start()

for t in threadler:
    t.join()

sure_4_thread = (time.time() - t4_basla) * 1000

print(f"1 Thread Calisma Suresi: {sure_1_thread:.4f} ms")
print(f"4 Thread Calisma Suresi: {sure_4_thread:.4f} ms")