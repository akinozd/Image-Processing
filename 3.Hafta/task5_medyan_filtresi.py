import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

h, w = img.shape
total_pixels = h * w

# 1. Salt-and-Pepper (Tuz ve Biber) gürültüsü üret
sp_noisy_img = img.copy()
salt_pepper_ratio = 0.05
num_noise = int(salt_pepper_ratio * total_pixels)

# Tuz (Beyaz noktalar)
coords_salt = [np.random.randint(0, i, num_noise // 2) for i in img.shape]
sp_noisy_img[tuple(coords_salt)] = 255

# Biber (Siyah noktalar)
coords_pepper = [np.random.randint(0, i, num_noise // 2) for i in img.shape]
sp_noisy_img[tuple(coords_pepper)] = 0

cv2.imwrite("cikti_task5_salt_pepper_gurultulu.jpg", sp_noisy_img)

# 2. OpenCV hazır medyan filtresi (5x5)
median_hazir = cv2.medianBlur(sp_noisy_img, ksize=5)
cv2.imwrite("cikti_task5_medyan_hazir.jpg", median_hazir)

# 3. Manuel 5x5 Medyan Filtresi
padded_sp = np.pad(sp_noisy_img, pad_width=2, mode='edge')
median_manuel = np.zeros((h, w), dtype=np.uint8)

for r in range(h):
    for c in range(w):
        # 5x5 penceredeki 25 değeri al ve sırala
        pencere = padded_sp[r : r + 5, c : c + 5].flatten()
        pencere.sort()
        # Ortadaki değer medyan (12. indeks)
        median_manuel[r, c] = pencere[12]

cv2.imwrite("cikti_task5_medyan_manuel.jpg", median_manuel)
print("Task 5 tamamlandı: 'cikti_task5_medyan_hazir.jpg' ve 'cikti_task5_medyan_manuel.jpg' kaydedildi.")