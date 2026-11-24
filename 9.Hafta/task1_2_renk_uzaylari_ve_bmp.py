import os
import cv2

# Resmi ac ve 1000x1000 boyutunda BMP olarak kaydet
img = cv2.imread("input.jpg")
renkli = cv2.resize(img, (1000, 1000))
cv2.imwrite("cikti_renkli.bmp", renkli)

# Griye cevir ve BMP kaydet
gri = cv2.cvtColor(renkli, cv2.COLOR_BGR2GRAY)
cv2.imwrite("cikti_gri.bmp", gri)

# Ekranda goster (herhangi bir tusa basinca kapanir)
cv2.imshow("Renkli 1000x1000", cv2.resize(renkli, (500, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Disk boyutlarini kiyasla
b_renkli = os.path.getsize("cikti_renkli.bmp")
b_gri = os.path.getsize("cikti_gri.bmp")

print(f"Renkli BMP (3 kanal): {b_renkli} Byte")
print(f"Gri BMP (1 kanal)   : {b_gri} Byte")
print(f"Oran: {b_renkli / b_gri:.2f} kat")