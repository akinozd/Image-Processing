import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("input.jpg bulunamadı!")

# 1. OpenCV hazır fonksiyonu ile CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe_cv2 = clahe.apply(img)
cv2.imwrite("cikti_task3_clahe_hazir.jpg", img_clahe_cv2)

# 2. Manuel CLAHE
def manuel_clahe(gorsel, clip_limit=2.0, grid_size=(8, 8)):
    h, w = gorsel.shape
    gh, gw = grid_size
    step_y = h // gh
    step_x = w // gw
    cikti = np.zeros_like(gorsel)
    
    for i in range(gh):
        for j in range(gw):
            y_start, y_end = i * step_y, (i + 1) * step_y if i != gh - 1 else h
            x_start, x_end = j * step_x, (j + 1) * step_x if j != gw - 1 else w
            
            tile = gorsel[y_start:y_end, x_start:x_end].copy()
            tile_h, tile_w = tile.shape
            n_pixels = tile_h * tile_w
            
            hist = [0] * 256
            for r in range(tile_h):
                for c in range(tile_w):
                    hist[tile[r, c]] += 1
            
            # Clipping
            clip_th = int(clip_limit * (n_pixels / 256.0))
            excess = 0
            for idx in range(256):
                if hist[idx] > clip_th:
                    excess += hist[idx] - clip_th
                    hist[idx] = clip_th
            
            bonus = excess // 256
            for idx in range(256):
                hist[idx] += bonus
                
            local_cdf = [0] * 256
            acc = 0
            for idx in range(256):
                acc += hist[idx]
                local_cdf[idx] = acc
                
            local_cdf_min = next((v for v in local_cdf if v > 0), 0)
            
            lut = np.zeros(256, dtype=np.uint8)
            diff = n_pixels - local_cdf_min
            for v in range(256):
                if diff > 0:
                    mapped = round(((local_cdf[v] - local_cdf_min) / diff) * 255)
                    lut[v] = np.clip(mapped, 0, 255)
                else:
                    lut[v] = v
                    
            for r in range(tile_h):
                for c in range(tile_w):
                    tile[r, c] = lut[tile[r, c]]
                    
            cikti[y_start:y_end, x_start:x_end] = tile
            
    return cikti

img_clahe_manuel = manuel_clahe(img, clip_limit=2.0, grid_size=(8, 8))
cv2.imwrite("cikti_task3_clahe_manuel.jpg", img_clahe_manuel)
print("Task 3 tamamlandı: 'cikti_task3_clahe_hazir.jpg' ve 'cikti_task3_clahe_manuel.jpg' kaydedildi.")