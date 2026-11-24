import cv2
import numpy as np
import matplotlib.pyplot as plt

def manuel_dft_2d(resim):
    M, N = resim.shape
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    W_M = np.exp(-2j * np.pi * u * x / M)

    v = np.arange(N).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)
    W_N = np.exp(-2j * np.pi * v * y / N)
    return W_M @ resim @ W_N

def manuel_idft_2d(harmonik):
    M, N = harmonik.shape
    u = np.arange(M).reshape(-1, 1)
    x = np.arange(M).reshape(1, -1)
    W_M_inv = np.exp(2j * np.pi * u * x / M)

    v = np.arange(N).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)
    W_N_inv = np.exp(2j * np.pi * v * y / N)

    resim = (W_M_inv @ harmonik @ W_N_inv) / (M * N)
    return np.clip(np.real(resim), 0, 255).astype(np.uint8)

def gauss_gurultu_ekle(resim, sigma=25):
    gurultulu = resim.astype(np.float64) + np.random.normal(0, sigma, resim.shape)
    return np.clip(gurultulu, 0, 255).astype(np.uint8)

# 10. Gorev: Low-Pass, High-Pass, Band-Pass ve Band-Reject Frekans Filtreleri
def frekans_filtresi_uygula(resim, filtre_tipi, r1=20, r2=45):
    M, N = resim.shape
    F = manuel_dft_2d(resim)
    F_shift = np.roll(np.roll(F, M // 2, axis=0), N // 2, axis=1)

    y, x = np.ogrid[:M, :N]
    mesafe = np.sqrt((x - N // 2) ** 2 + (y - M // 2) ** 2)

    if filtre_tipi == "low_pass":
        maske = mesafe <= r1
    elif filtre_tipi == "high_pass":
        maske = mesafe > r1
    elif filtre_tipi == "band_pass":
        maske = (mesafe >= r1) & (mesafe <= r2)
    elif filtre_tipi == "band_reject":
        maske = (mesafe < r1) | (mesafe > r2)
    else:
        maske = np.ones((M, N), dtype=bool)

    F_filtreli_shift = F_shift * maske
    F_filtreli = np.roll(np.roll(F_filtreli_shift, -(M // 2), axis=0), -(N // 2), axis=1)
    return manuel_idft_2d(F_filtreli)

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))
img_gurultulu = gauss_gurultu_ekle(img, sigma=25)

filtreler = ["low_pass", "high_pass", "band_pass", "band_reject"]
basliklar = ["Low-Pass", "High-Pass", "Band-Pass", "Band-Reject"]

plt.figure(figsize=(14, 7))

# Ust satir: Orijinal goruntu uzerinde filtreler
plt.subplot(2, 5, 1)
plt.imshow(img, cmap="gray")
plt.title("Orijinal")
plt.axis("off")

for i in range(4):
    sonuc = frekans_filtresi_uygula(img, filtreler[i])
    plt.subplot(2, 5, i + 2)
    plt.imshow(sonuc, cmap="gray")
    plt.title(f"Orj + {basliklar[i]}")
    plt.axis("off")

# Alt satir: Gurultulu goruntu uzerinde filtreler
plt.subplot(2, 5, 6)
plt.imshow(img_gurultulu, cmap="gray")
plt.title("Gurultulu")
plt.axis("off")

for i in range(4):
    sonuc_g = frekans_filtresi_uygula(img_gurultulu, filtreler[i])
    plt.subplot(2, 5, i + 7)
    plt.imshow(sonuc_g, cmap="gray")
    plt.title(f"Gurultu + {basliklar[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task10_frekans_filtreleri.png")

print("Task 10 bitti.")
print("\n--- FREKANS FILTRELERI ETKI RAPORU (TASK 10) ---")
print("1. Low-Pass (Alcak Geciren): Yuksek frekansli gurultuleri temizler, genel parlakligi (DC bilesenini) korur ancak kenar gecislerini bulaniklastirir.")
print("2. High-Pass (Yuksek Geciren): Merkezdeki DC bilesenini kestigi icin genel parlaklik siyaha duser; sadece keskin kenar gecislerini ve yuksek frekansli gurultuleri one cikarir.")
print("3. Band-Pass (Bant Geciren): Hem cok dusuk frekansli parlakligi hem de cok yuksek frekansli ince gurultuyu eler; belirli kalinliktaki doku ve orta olcekli kenarlari gosterir.")
print("4. Band-Reject (Bant Durduran): Genel parlakligi ve ana hatlari korurken sadece belirli bir frekans bandindaki periyodik gurultuleri bastirir.")