import cv2
import numpy as np
import matplotlib.pyplot as plt

# 7. Gorev: 2-Boyutlu Manuel DFT fonksiyonu (Satir ve sutun bazli matris carpimi ile hizli 2D DFT)
def manuel_dft_2d(resim):
    M, N = resim.shape
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    W_M = np.exp(-2j * np.pi * u * x / M)

    v = np.arange(N).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)
    W_N = np.exp(-2j * np.pi * v * y / N)

    # 2D DFT: F = W_M * f * W_N
    harmonik_matris = W_M @ resim @ W_N
    return harmonik_matris

# Spektrumu gorsellestirmek icin merkezleme (fftshift mantigi) ve log olcekleme
def spektrum_goruntusu(harmonik):
    M, N = harmonik.shape
    kaydirilmis = np.roll(np.roll(harmonik, M // 2, axis=0), N // 2, axis=1)
    return np.log(1 + np.abs(kaydirilmis))

# 8. Gorev: Gauss ve Salt & Pepper gurultu ekleme fonksiyonlari
def gauss_gurultu_ekle(resim, sigma):
    gurultu = np.random.normal(0, sigma, resim.shape)
    gurultulu = resim.astype(np.float64) + gurultu
    return np.clip(gurultulu, 0, 255).astype(np.uint8)

def salt_and_pepper_ekle(resim, oran):
    gurultulu = resim.copy()
    toplam_piksel = resim.size
    # Tuz (beyaz - 255)
    tuz_sayisi = int(toplam_piksel * oran / 2)
    coords_tuz = [np.random.randint(0, i, tuz_sayisi) for i in resim.shape]
    gurultulu[coords_tuz[0], coords_tuz[1]] = 255
    # Biber (siyah - 0)
    biber_sayisi = int(toplam_piksel * oran / 2)
    coords_biber = [np.random.randint(0, i, biber_sayisi) for i in resim.shape]
    gurultulu[coords_biber[0], coords_biber[1]] = 0
    return gurultulu

# Resmi oku ve hizli 2D DFT icin 128x128 gri tonlamaya cevir
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))

# Gurultulu goruntuleri olustur
img_gauss = gauss_gurultu_ekle(img, sigma=25)
img_sp = salt_and_pepper_ekle(img, oran=0.08)

# 2D DFT harmonik matrislerini hesapla
dft_orj = manuel_dft_2d(img)
dft_gauss = manuel_dft_2d(img_gauss)
dft_sp = manuel_dft_2d(img_sp)

# Gorsellestirme
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Orijinal Resim (128x128)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_gauss, cmap="gray")
plt.title("Gauss Gurultulu (sigma=25)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_sp, cmap="gray")
plt.title("Salt & Pepper Gurultulu (oran=0.08)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(spektrum_goruntusu(dft_orj), cmap="gray")
plt.title("7. Gorev: Orijinal 2D DFT Spektrumu")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(spektrum_goruntusu(dft_gauss), cmap="gray")
plt.title("8. Gorev: Gauss 2D DFT Spektrumu")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(spektrum_goruntusu(dft_sp), cmap="gray")
plt.title("8. Gorev: Salt & Pepper 2D DFT Spektrumu")
plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task7_8_2d_dft_spektrum.png")
print("Task 7 ve 8 bitti.")
print("Kiyaslama Yorumu (Task 8): Orijinal resimde enerji merkezde (dusuk frekanslarda) toplanirken, Gauss ve Salt & Pepper gurultuleri spektrumun dis kenarlarina (yuksek frekans bolgelerine) homojen bir parlaklik/parazit ekler.")