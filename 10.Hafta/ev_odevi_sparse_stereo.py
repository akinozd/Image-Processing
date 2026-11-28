import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

sol = cv2.imread("scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE)
sag = cv2.imread("scene1.row3.col3.ppm", cv2.IMREAD_GRAYSCALE)
gt = cv2.imread("truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Tsukuba ground truth degerleri 16 ile olceklenmistir (0-255 araligini piksel kaymasina cevir)
gt_disparity = gt / 16.0

sol_f = sol.astype(np.float32)
sag_f = sag.astype(np.float32)

# 1. Canny ile kenar tespiti
kenarlar = cv2.Canny(sol, 80, 180) > 0

def sad_eslestirme(sol_img, sag_img, maske=None, blok=7, max_disp=24):
    h, w = sol_img.shape
    yari = blok // 2
    disp = np.zeros((h, w), dtype=np.float32)

    for y in range(yari, h - yari):
        for x in range(yari + max_disp, w - yari):
            if maske is not None and not maske[y, x]:
                continue

            blok_sol = sol_img[y - yari:y + yari + 1, x - yari:x + yari + 1]
            en_iyi_sad = 1e18
            en_iyi_d = 0

            for d in range(max_disp):
                blok_sag = sag_img[y - yari:y + yari + 1, x - d - yari:x - d + yari + 1]
                sad = np.sum(np.abs(blok_sol - blok_sag))
                if sad < en_iyi_sad:
                    en_iyi_sad = sad
                    en_iyi_d = d

            disp[y, x] = en_iyi_d
    return disp

# Seyrek (Sparse) Stereo: Sadece kenar pikselleri
t0 = time.perf_counter()
disp_sparse = sad_eslestirme(sol_f, sag_f, maske=kenarlar)
sure_sparse = time.perf_counter() - t0

# Yogun (Dense) Stereo: Tum pikseller
t1 = time.perf_counter()
disp_dense = sad_eslestirme(sol_f, sag_f, maske=None)
sure_dense = time.perf_counter() - t1

# Dogruluk kiyasi (Sadece gecerli kenar bolgeleri uzerinden Ortalama Mutlak Hata - MAE)
gecerli = kenarlar & (gt_disparity > 0)
gecerli[:, :27] = False

mae_sparse = np.mean(np.abs(disp_sparse[gecerli] - gt_disparity[gecerli]))
mae_dense = np.mean(np.abs(disp_dense[gecerli] - gt_disparity[gecerli]))

print("--- SPARSE vs DENSE STEREO KIYASLAMASI ---")
print(f"Sparse Stereo Sure: {sure_sparse:.2f} sn | Kenar Noktasi MAE Hatasi: {mae_sparse:.2f} piksel")
print(f"Dense Stereo Sure : {sure_dense:.2f} sn | Tum Piksel MAE Hatasi  : {mae_dense:.2f} piksel")
print(f"Hiz Kazanci       : {sure_dense / sure_sparse:.1f} kat daha hizli")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(kenarlar, cmap="gray")
plt.title("Canny Kenar Haritasi")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(disp_sparse, cmap="jet")
plt.title(f"Sparse Stereo ({sure_sparse:.2f} sn)\nMAE: {mae_sparse:.2f} px")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(disp_dense, cmap="jet")
plt.title(f"Dense Stereo ({sure_dense:.2f} sn)\nMAE: {mae_dense:.2f} px")
plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_ev_odevi_sparse_vs_dense.png")