import cv2
import glob
import os

girdi_yolu = "inputs/*.jpg"
resimler = sorted(glob.glob(girdi_yolu))

if not resimler:
    print("inputs klasorunde gorsel bulunamadi!")
    exit()

hedef_boyut = (1024, 768)

for i, yol in enumerate(resimler, start=1):
    img = cv2.imread(yol)
    if img is None:
        continue
    
    #INTER_LINEAR
    boyutlandirilmis = cv2.resize(img, hedef_boyut, interpolation=cv2.INTER_LINEAR)
    
    cikti_adi = f"cikti_{i}_1024x768.jpg"
    cv2.imwrite(cikti_adi, boyutlandirilmis)
    print(f"{yol} -> {cikti_adi} olarak kaydedildi.")

print("Tum gorseller 1024x768 boyutuna getirildi.")