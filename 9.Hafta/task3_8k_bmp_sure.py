import time
import cv2
import numpy as np

# 1. 8K (7680x4320) 3-kanalli BMP olustur ve kaydet
img_8k = np.random.randint(0, 256, (4320, 7680, 3), dtype=np.uint8)
cv2.imwrite("input_8k.bmp", img_8k)

# 2. Olusturulan 8K BMP resmini ac
acilan = cv2.imread("input_8k.bmp")

# 3. Sadece piksel degerlerini 2'ye bolme kisminin suresini hassas olc
baslangic = time.perf_counter()
duzenlenmis = (acilan // 2).astype(np.uint8)
bitis = time.perf_counter()

# 4. Duzenlenen resmi BMP olarak kaydet
cv2.imwrite("cikti_task3_8k_yarim.bmp", duzenlenmis)

print(f"8K Piksel Duzenleme Suresi: {bitis - baslangic:.6f} saniye")