import cv2
import numpy as np
import matplotlib.pyplot as plt

#Kendi calcHist fonksiyonumuz
def manuel_calchist(kanal_verisi):
    hist = np.zeros(256, dtype=int)
    # Kanal matrisini tek boyuta indirip sayaçları artırıyoruz
    for piksel in kanal_verisi.flatten():
        hist[piksel] += 1
    return hist

# Görseli oku
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

kanallar = ('b', 'g', 'r')
renkler = ('blue', 'green', 'red')

plt.figure(figsize=(8, 5))
plt.title("Task 2: Manuel calcHist ile Kanal Histogramlari")
plt.xlabel("Piksel Degeri (0-255)")
plt.ylabel("Piksel Sayisi")

#Her kanal için fonksiyonu çalıştır
for i, col in enumerate(kanallar):
    # İlgili kanalı al 
    kanal = img[:, :, i]
    
    # Kendi fonksiyonumuz ile hesaplama
    bizim_hist = manuel_calchist(kanal)
    
    # openCV'nin hazır fonksiyonu ile doğrulama
    cv_hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
    
    # Hata kontrolü
    fark = np.sum(np.abs(bizim_hist - cv_hist))
    print(f"{col.upper()} Kanali icin Manuel vs OpenCV Mutlak Fark: {fark}")
    
    plt.plot(bizim_hist, color=renkler[i], label=f"{col.upper()} Kanali (Manuel)")

plt.xlim([0, 256])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Görsel çıktıyı kaydet
plt.savefig("cikti_task2_manuel_histogram.png")
print("Task 2 tamamlandi: 'cikti_task2_manuel_histogram.png' kaydedildi.")