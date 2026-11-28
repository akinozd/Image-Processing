import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

sol = cv2.imread("scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE).astype(np.float32)
sag = cv2.imread("scene1.row3.col3.ppm", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# 3. Gorev: Benzer parametre yapisiyla SAD, SSD ve ZNCC benzerlik fonksiyonlari
def benzerlik_sad(blok_sol, blok_sag):
    return -np.sum(np.abs(blok_sol - blok_sag))  # Kucuk fark yuksek skor olmasi icin eksi

def benzerlik_ssd(blok_sol, blok_sag):
    return -np.sum((blok_sol - blok_sag) ** 2)

def benzerlik_zncc(blok_sol, blok_sag):
    sol_norm = blok_sol - np.mean(blok_sol)
    sag_norm = blok_sag - np.mean(blok_sag)
    payda = np.sqrt(np.sum(sol_norm ** 2) * np.sum(sag_norm ** 2)) + 1e-6
    return np.sum(sol_norm * sag_norm) / payda

# 2. Gorev: OpenCV kullanmadan manuel blok eslestirme
def manuel_stereo_eslestirme(sol_img, sag_img, benzerlik_fonk, blok_boyutu=7, max_disp=24):
    h, w = sol_img.shape
    yari = blok_boyutu // 2
    disparity = np.zeros((h, w), dtype=np.float32)

    for y in range(yari, h - yari):
        for x in range(yari + max_disp, w - yari):
            blok_sol = sol_img[y - yari:y + yari + 1, x - yari:x + yari + 1]
            en_iyi_skor = -1e18
            en_iyi_d = 0

            for d in range(max_disp):
                blok_sag = sag_img[y - yari:y + yari + 1, x - d - yari:x - d + yari + 1]
                skor = benzerlik_fonk(blok_sol, blok_sag)
                if skor > en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_d = d

            disparity[y, x] = en_iyi_d
    return disparity

yontemler = [
    ("SAD (Mutlak Fark)", benzerlik_sad),
    ("SSD (Karesel Fark)", benzerlik_ssd),
    ("ZNCC (Normalize Korelasyon)", benzerlik_zncc)
]

plt.figure(figsize=(15, 5))

for i, (isim, fonk) in enumerate(yontemler):
    t0 = time.perf_counter()
    disp = manuel_stereo_eslestirme(sol, sag, fonk, blok_boyutu=7, max_disp=24)
    sure = time.perf_counter() - t0
    print(f"{isim} tamamlandi -> Sure: {sure:.2f} sn")

    plt.subplot(1, 3, i + 1)
    plt.imshow(disp, cmap="jet")
    plt.title(f"{isim}\n({sure:.2f} sn)")
    plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task2_3_sad_ssd_zncc.png")