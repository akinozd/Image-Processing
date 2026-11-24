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

# 9. Gorev: 2-Boyutlu Manuel Ters DFT (IDFT)
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

# Zero Mean Normalized Cross Correlation (ZNCC) Metrigi [-1, 1]
def zncc_hesapla(img1, img2):
    f = img1.astype(np.float64) - np.mean(img1)
    g = img2.astype(np.float64) - np.mean(img2)
    pay = np.sum(f * g)
    payda = np.sqrt(np.sum(f ** 2) * np.sum(g ** 2))
    return pay / payda if payda != 0 else 0

def gauss_gurultu_ekle(resim, sigma=25):
    gurultulu = resim.astype(np.float64) + np.random.normal(0, sigma, resim.shape)
    return np.clip(gurultulu, 0, 255).astype(np.uint8)

def salt_and_pepper_ekle(resim, oran=0.08):
    gurultulu = resim.copy()
    tuz = int(resim.size * oran / 2)
    c1 = [np.random.randint(0, i, tuz) for i in resim.shape]
    gurultulu[c1[0], c1[1]] = 255
    c2 = [np.random.randint(0, i, tuz) for i in resim.shape]
    gurultulu[c2[0], c2[1]] = 0
    return gurultulu

# Frekans uzayinda merkez (dusuk frekans) haric gurultuyu temizleyip IDFT ile donus
def frekanstan_temizle_ve_idft(gurultulu_resim, yaricap=35):
    M, N = gurultulu_resim.shape
    F = manuel_dft_2d(gurultulu_resim)
    F_shift = np.roll(np.roll(F, M // 2, axis=0), N // 2, axis=1)

    y, x = np.ogrid[:M, :N]
    maske = ((x - N // 2) ** 2 + (y - M // 2) ** 2) <= yaricap ** 2
    F_temiz_shift = F_shift * maske

    F_temiz = np.roll(np.roll(F_temiz_shift, -(M // 2), axis=0), -(N // 2), axis=1)
    return manuel_idft_2d(F_temiz)

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))

img_gauss = gauss_gurultu_ekle(img, sigma=25)
img_sp = salt_and_pepper_ekle(img, oran=0.08)

rec_gauss = frekanstan_temizle_ve_idft(img_gauss, yaricap=35)
rec_sp = frekanstan_temizle_ve_idft(img_sp, yaricap=35)

# ZNCC Skorlari
zncc_gauss_once = zncc_hesapla(img, img_gauss)
zncc_gauss_sonra = zncc_hesapla(img, rec_gauss)
zncc_sp_once = zncc_hesapla(img, img_sp)
zncc_sp_sonra = zncc_hesapla(img, rec_sp)

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(img_gauss, cmap="gray")
plt.title(f"Gauss Gurultulu (ZNCC: {zncc_gauss_once:.3f})")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(rec_gauss, cmap="gray")
plt.title(f"IDFT Sonrasi (ZNCC: {zncc_gauss_sonra:.3f})")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img_sp, cmap="gray")
plt.title(f"Salt & Pepper (ZNCC: {zncc_sp_once:.3f})")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(rec_sp, cmap="gray")
plt.title(f"IDFT Sonrasi (ZNCC: {zncc_sp_sonra:.3f})")
plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task9_idft_zncc.png")

print("Task 9 bitti.")
print(f"Gauss ZNCC -> Oncesi: {zncc_gauss_once:.4f} | IDFT Sonrasi: {zncc_gauss_sonra:.4f}")
print(f"S&P   ZNCC -> Oncesi: {zncc_sp_once:.4f} | IDFT Sonrasi: {zncc_sp_sonra:.4f}")
print("\n--- BENZERLIK METRIKLERI RAPORU (TASK 9) ---")
print("1. ZNCC (Zero Mean Normalized Cross Correlation): [-1, 1] araligindadir. Ortalamayi cikarip standart sapmaya boldugu icin parlaklik ve kontrast degisimlerinden etkilenmez. Dezavantaji: Hesaplama maliyeti MSE'ye gore yuksektir.")
print("2. MSE / PSNR (Mean Squared Error / Peak SNR): Piksel farklarinin karesini alir, hesaplamasi cok hizlidir. Dezavantaji: Kucuk parlaklik kaymalarinda bile yuksek hata verir, insan gozunun yapisal algisini yansitmaz.")
print("3. SSIM (Structural Similarity Index): Parlaklik, kontrast ve yapisal benzerligi birlikte olcer; insan algisina en yakin sonuclari verir. Dezavantaji: Parametre ve pencere boyutuna bagimlidir, ZNCC ve MSE'den yavastir.")