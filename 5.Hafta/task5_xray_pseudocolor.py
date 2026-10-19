import cv2
import os
import glob
import numpy as np

input_dir = "xray_inputs"
output_dir = "xray_outputs"
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
if len(image_paths) == 0:
    # uzantı .png veya başka bir formatta kaydedildi mi diye hepsini tara
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))

if len(image_paths) == 0:
    print(f"Hata: '{input_dir}' klasorunde gorsel bulunamadi!")
    exit()

print(f"Toplam {len(image_paths)} adet X-Ray gorseli isleniyor...\n")

for idx, path in enumerate(image_paths[:5], start=1):
    original_bgr = cv2.imread(path)
    if original_bgr is None:
        continue
    
    # Gri tonlamaya dönüştür
    gray_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # Güvenlik operatörleri için sahte renklendirme (Pseudocoloring)
    # COLORMAP_JET / COLORMAP_TURBO paleti X-ışını soğurma yoğunluğuna göre
    # mavi (düşük yoğunluk) -> sarı -> kırmızı (yoğun metal nesneler) kontrastı sağlar.
    colored_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_TURBO)
    
    # Kıyaslama paneli: Yan yana Orijinal, Gri, Sahte Renkli
    gray_3c = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    panel = np.hstack((original_bgr, gray_3c, colored_img))
    
    # kaydet
    cv2.imwrite(os.path.join(output_dir, f"xray_{idx}_gri.jpg"), gray_img)
    cv2.imwrite(os.path.join(output_dir, f"xray_{idx}_renkli.jpg"), colored_img)
    cv2.imwrite(os.path.join(output_dir, f"xray_{idx}_panel.jpg"), panel)
    
    print(f"[{idx}/5] {os.path.basename(path)} tamamlandı -> Gri ve Renkli kaydedildi.")

print(f"\nTask 5 tamamlandi: Tüm ciktilar '{output_dir}' altina kaydedildi.")