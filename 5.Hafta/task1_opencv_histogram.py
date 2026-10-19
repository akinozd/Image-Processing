import cv2
import matplotlib.pyplot as plt

# Renkli görüntüyü oku (Varsayılan BGR formatında gelir)
img = cv2.imread("input.jpg")
if img is None:
    raise FileNotFoundError("input.jpg bulunamadi!")

# Kanalların isimleri ve OpenCV BGR sıra indeksleri
kanallar = ('b', 'g', 'r')
renkler = ('blue', 'green', 'red')

plt.figure(figsize=(8, 5))
plt.title("Task 1: OpenCV ile Kanal Histogramlari")
plt.xlabel("Piksel Degeri (0-255)")
plt.ylabel("Piksel Sayisi")

# Her kanal için ayrı ayrı calcHisrt çalıştır ve çizdir
for i, col in enumerate(kanallar):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=renkler[i], label=f"{col.upper()} Kanali")

plt.xlim([0, 256])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# görsel çıktıyı kaydet
plt.savefig("cikti_task1_opencv_histogram.png")
print("Task 1 tamamlandi: 'cikti_task1_opencv_histogram.png' kaydedildi.")