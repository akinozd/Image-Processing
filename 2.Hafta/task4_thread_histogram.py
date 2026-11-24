import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

yukseklik, genislik = img.shape
parca_boyutu = yukseklik // 4

# Veri cakismasi (race condition) olmamasi icin her thread kendi dizisine yazacak
yerel_histogramlar = [np.zeros(256, dtype=np.int64) for _ in range(4)]

def parca_histogram_hesapla(resim, bas_satir, bit_satir, thread_id):
    hist = np.zeros(256, dtype=np.int64)
    for i in range(bas_satir, bit_satir):
        for j in range(genislik):
            piksel_degeri = resim[i, j]
            hist[piksel_degeri] += 1
    yerel_histogramlar[thread_id] = hist

threadler = []
for i in range(4):
    bas = i * parca_boyutu
    son = yukseklik if i == 3 else (i + 1) * parca_boyutu
    t = threading.Thread(target=parca_histogram_hesapla, args=(img, bas, son, i))
    threadler.append(t)
    t.start()

for t in threadler:
    t.join()

# 4 yerel histogrami toplayarak ana histogrami elde etme (Reduction)
ana_histogram = np.zeros(256, dtype=np.int64)
for i in range(4):
    ana_histogram += yerel_histogramlar[i]

print(f"Resimdeki toplam piksel sayisi: {yukseklik * genislik}")
print(f"Birlestirilen histogram toplami: {np.sum(ana_histogram)}")

plt.figure(figsize=(8, 4))
plt.title("Task 4: 4 Thread ile Paralel Hesaplanan Ana Histogram")
plt.xlabel("Piksel Degeri (0-255)")
plt.ylabel("Piksel Sayisi")
plt.plot(ana_histogram, color="black")
plt.xlim([0, 256])
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("cikti_task4_histogram.png")
print("Task 4 tamamlandi: 'cikti_task4_histogram.png' kaydedildi.")