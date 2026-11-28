import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tsukuba gercek dosyalarini yukle
sol = cv2.imread("scene1.row3.col1.ppm", cv2.IMREAD_GRAYSCALE)
sag = cv2.imread("scene1.row3.col3.ppm", cv2.IMREAD_GRAYSCALE)
gercek_disp = cv2.imread("truedisp.row3.col3.pgm", cv2.IMREAD_GRAYSCALE)

# 1. StereoBM (Block Matching) ile Disparity Hesaplama
stereo_bm = cv2.StereoBM_create(numDisparities=32, blockSize=15)
disp_bm = stereo_bm.compute(sol, sag).astype(np.float32) / 16.0

# 2. StereoSGBM (Semi-Global Block Matching) ile Disparity Hesaplama
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=32,
    blockSize=5,
    P1=8 * 3 * 5 ** 2,
    P2=32 * 3 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disp_sgbm = stereo_sgbm.compute(sol, sag).astype(np.float32) / 16.0

# Gorsellestirme ve Kayit
plt.figure(figsize=(14, 7))

plt.subplot(2, 3, 1)
plt.imshow(sol, cmap="gray")
plt.title("Sol Goruntu (col1)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(sag, cmap="gray")
plt.title("Sag Goruntu (col3)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gercek_disp, cmap="jet")
plt.title("Gercek Derinlik (Ground Truth)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(disp_bm, cmap="jet")
plt.title("StereoBM Disparity")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(disp_sgbm, cmap="jet")
plt.title("StereoSGBM Disparity")
plt.axis("off")

plt.tight_layout()
plt.savefig("cikti_task1_stereo_disparity.png")
print("10. Hafta Tsukuba Stereo Eslestirme bitti.")