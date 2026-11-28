import cv2
import numpy as np
import matplotlib.pyplot as plt

sol = cv2.imread("scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE).astype(np.float32)
sag = cv2.imread("scene1.row3.col3.ppm", cv2.IMREAD_GRAYSCALE).astype(np.float32)

def maliyet_hacmi_hesapla(sol_img, sag_img, max_disp=24, blok=5):
    h, w = sol_img.shape
    dsi = np.full((h, w, max_disp), 1e5, dtype=np.float32)
    kernel = np.ones((blok, blok), np.float32)
    for d in range(max_disp):
        kaydirilmis_sag = np.zeros_like(sag_img)
        kaydirilmis_sag[:, d:] = sag_img[:, :w - d]
        fark = np.abs(sol_img - kaydirilmis_sag)
        dsi[:, :, d] = cv2.filter2D(fark, -1, kernel)
    return dsi

def yol_maliyeti_topla(maliyet_2d, P1, P2):
    uzunluk, max_disp = maliyet_2d.shape
    L = np.zeros_like(maliyet_2d)
    L[0] = maliyet_2d[0]
    d_idx = np.arange(max_disp)

    for i in range(1, uzunluk):
        onceki = L[i - 1]
        min_onceki = np.min(onceki)
        # Komsu pikseller arasi disparity degisimi icin smoothness cezalari (P1 ve P2)
        fark_matrisi = np.abs(d_idx[:, None] - d_idx[None, :])
        ceza = np.where(fark_matrisi == 0, 0, np.where(fark_matrisi == 1, P1, P2))
        en_ucuz_gecis = np.min(onceki[None, :] + ceza, axis=1)
        L[i] = maliyet_2d[i] + en_ucuz_gecis - min_onceki
    return L

def manuel_sgbm(sol_img, sag_img, P1, P2, max_disp=24):
    dsi = maliyet_hacmi_hesapla(sol_img, sag_img, max_disp=max_disp)
    h, w, _ = dsi.shape
    toplam_maliyet = np.zeros_like(dsi)

    # 4 ana yonde (Yatay ve Dikey) dinamik programlama ile maliyet biriktirme
    for y in range(h):
        toplam_maliyet[y] += yol_maliyeti_topla(dsi[y], P1, P2)
        toplam_maliyet[y] += yol_maliyeti_topla(dsi[y, ::-1], P1, P2)[::-1]

    for x in range(w):
        toplam_maliyet[:, x] += yol_maliyeti_topla(dsi[:, x], P1, P2)
        toplam_maliyet[:, x] += yol_maliyeti_topla(dsi[::-1, x], P1, P2)[::-1]

    return np.argmin(toplam_maliyet, axis=2).astype(np.float32)

# Dusuk ve Yuksek Smoothness (P1, P2) karsilastirmasi
disp_dusuk_smooth = manuel_sgbm(sol, sag, P1=10, P2=40)
disp_yuksek_smooth = manuel_sgbm(sol, sag, P1=150, P2=600)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(disp_dusuk_smooth, cmap="jet")
plt.title("Dusuk Smoothness (P1=10, P2=40)\nDaha gurultulu, keskin gecisler")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(disp_yuksek_smooth, cmap="jet")
plt.title("Yuksek Smoothness (P1=150, P2=600)\nPuruzsuz yuzeyler, yumusak gecis")
plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task4_manuel_sgbm.png")
print("Task 4 Manuel SGBM bitti.")