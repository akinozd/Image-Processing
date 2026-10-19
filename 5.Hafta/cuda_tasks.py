import cv2
import numpy as np

# CUDA kütüphanesi kontrolü (CuPy / Numba)
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Bilgi: CuPy kurulu degil veya GPU bulunamadi. GPU simülasyonu mantigiyla calisilacak.")

# -------------------------------------------------------------
# Task 1 & 2: CUDA ile Kanal Histogrami (Atomic Add / GPU Mantigi)
# -------------------------------------------------------------
def cuda_histogram(img_bgr):
    if not CUDA_AVAILABLE:
        print("[CUDA Task 1-2] GPU atlandi.")
        return None
    
    # Görüntüyü GPU belleğine aktar
    d_img = cp.asarray(img_bgr)
    histograms = []
    
    # Her kanal için GPU üzerinde histogram hesabı
    for i in range(3):
        d_channel = d_img[:, :, i]
        # CuPy dahili CUDA kernel'i bincount kullanır
        d_hist = cp.bincount(d_channel.ravel(), minlength=256)
        # Sonucu CPU belleğine geri al
        histograms.append(cp.asnumpy(d_hist))
        
    print("[CUDA Task 1-2] Kanal histogramlari GPU uzerinde hesaplandi.")
    return histograms

# -------------------------------------------------------------
# Task 3: CUDA ile RGB -> Gray Donusumu (BT.601 Paralel Kernel)
# -------------------------------------------------------------
def cuda_bgr2gray(img_bgr):
    if not CUDA_AVAILABLE:
        print("[CUDA Task 3] GPU atlandi.")
        return None
    
    d_img = cp.asarray(img_bgr, dtype=cp.float32)
    b = d_img[:, :, 0]
    g = d_img[:, :, 1]
    r = d_img[:, :, 2]
    
    # GPU çekirdekleri tüm piksellerde ayni anda paralel calisir:
    # Y = 0.299*R + 0.587*G + 0.114*B
    d_gray = 0.299 * r + 0.587 * g + 0.114 * b
    d_gray = cp.clip(cp.round(d_gray), 0, 255).astype(cp.uint8)
    
    gray_cpu = cp.asnumpy(d_gray)
    cv2.imwrite("cikti_cuda_task3_gray.jpg", gray_cpu)
    print("[CUDA Task 3] Gri donusum GPU uzerinde tamamlandi: 'cikti_cuda_task3_gray.jpg'")
    return gray_cpu

# -------------------------------------------------------------
# Task 4: CUDA ile BGR -> YCrCb Donusumu (Paralel Matris Matrisi)
# -------------------------------------------------------------
def cuda_bgr2ycrcb(img_bgr):
    if not CUDA_AVAILABLE:
        print("[CUDA Task 4] GPU atlandi.")
        return None
    
    d_img = cp.asarray(img_bgr, dtype=cp.float32)
    b = d_img[:, :, 0]
    g = d_img[:, :, 1]
    r = d_img[:, :, 2]
    
    # GPU paralel formul hesaplamalari
    d_y  = 0.299 * r + 0.587 * g + 0.114 * b
    d_cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    d_cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    
    d_y  = cp.clip(cp.round(d_y),  0, 255).astype(cp.uint8)
    d_cr = cp.clip(cp.round(d_cr), 0, 255).astype(cp.uint8)
    d_cb = cp.clip(cp.round(d_cb), 0, 255).astype(cp.uint8)
    
    d_ycrcb = cp.dstack([d_y, d_cr, d_cb])
    ycrcb_cpu = cp.asnumpy(d_ycrcb)
    
    cv2.imwrite("cikti_cuda_task4_ycrcb.jpg", ycrcb_cpu)
    print("[CUDA Task 4] YCrCb donusumu GPU uzerinde tamamlandi: 'cikti_cuda_task4_ycrcb.jpg'")
    return ycrcb_cpu

# -------------------------------------------------------------
# Task 5: CUDA ile X-Ray Sahte Renklendirme (GPU Lookup Table)
# -------------------------------------------------------------
def cuda_pseudocolor(xray_path):
    if not CUDA_AVAILABLE:
        print("[CUDA Task 5] GPU atlandi.")
        return
    
    img = cv2.imread(xray_path)
    if img is None:
        return
    
    # 1. GPU'da griye cevir
    d_img = cp.asarray(img, dtype=cp.float32)
    d_gray = 0.299 * d_img[:, :, 2] + 0.587 * d_img[:, :, 1] + 0.114 * d_img[:, :, 0]
    d_gray = cp.clip(cp.round(d_gray), 0, 255).astype(cp.uint8)
    
    # 2. Renk paleti tablosunu (LUT) GPU'ya aktar
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
    d_lut = cp.asarray(lut)  # Shape: (256, 1, 3)
    
    # GPU paralel indeksleme (Lookup)
    d_colored = d_lut[d_gray, 0, :]
    
    colored_cpu = cp.asnumpy(d_colored)
    cv2.imwrite("cikti_cuda_task5_xray_colored.jpg", colored_cpu)
    print("[CUDA Task 5] X-Ray Sahte Renklendirme GPU'da tamamlandi: 'cikti_cuda_task5_xray_colored.jpg'")

# -------------------------------------------------------------
# Ana Calistirma Blogu
# -------------------------------------------------------------
if __name__ == "__main__":
    img = cv2.imread("input.jpg")
    if img is not None and CUDA_AVAILABLE:
        cuda_histogram(img)
        cuda_bgr2gray(img)
        cuda_bgr2ycrcb(img)
        cuda_pseudocolor("xray_inputs/xray1.jpg")
    elif not CUDA_AVAILABLE:
        print("CUDA kutuphanesi bulunamadigi icin fonksiyonlar reponuza eklenmek uzere hazirlandi.")