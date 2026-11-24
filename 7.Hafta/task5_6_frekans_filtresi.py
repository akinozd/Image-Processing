import numpy as np
import matplotlib.pyplot as plt

def sinus_olustur(N, genlik, frekans, faz=0):
    t = np.arange(N) / N
    vektor = np.zeros(N)
    for i in range(N):
        vektor[i] = genlik * np.sin(2 * np.pi * frekans * t[i] + faz)
    return vektor

def gurultu_ekle(vektor, gurultu_orani):
    genlik = np.max(np.abs(vektor))
    gurultulu = np.zeros(len(vektor))
    for i in range(len(vektor)):
        gurultulu[i] = vektor[i] + np.random.uniform(-1, 1) * genlik * gurultu_orani
    return gurultulu

# Manuel 1D DFT
def manuel_dft_1d(vektor):
    N = len(vektor)
    harmonik = np.zeros(N, dtype=complex)
    for k in range(N):
        toplam = 0j
        for n in range(N):
            aci = -2 * np.pi * k * n / N
            toplam += vektor[n] * (np.cos(aci) + 1j * np.sin(aci))
        harmonik[k] = toplam
    return harmonik

# Manuel 1D Ters DFT (IDFT)
def manuel_idft_1d(harmonik):
    N = len(harmonik)
    sinyal = np.zeros(N, dtype=complex)
    for n in range(N):
        toplam = 0j
        for k in range(N):
            aci = 2 * np.pi * k * n / N
            toplam += harmonik[k] * (np.cos(aci) + 1j * np.sin(aci))
        sinyal[n] = toplam / N
    return np.real(sinyal)

# 6. Gorev: Esik degerin altindaki harmonikleri temizleyip sinyali yeniden ureten Frekans Filtresi
def frekans_filtresi(gurultulu_sinyal, esik_orani=0.25):
    harmonik = manuel_dft_1d(gurultulu_sinyal)
    genlikler = np.abs(harmonik)
    esik = np.max(genlikler) * esik_orani
    
    temiz_harmonik = np.zeros_like(harmonik)
    for k in range(len(harmonik)):
        if genlikler[k] >= esik:
            temiz_harmonik[k] = harmonik[k]
            
    yeniden_sinyal = manuel_idft_1d(temiz_harmonik)
    return yeniden_sinyal

# 5. Gorev: f=50, A=12 ve f=30, A=5 sinyalleri (N=256)
N = 256
s1_temiz = sinus_olustur(N, genlik=12, frekans=50)
s2_temiz = sinus_olustur(N, genlik=5, frekans=30)

s1_gurultulu = gurultu_ekle(s1_temiz, 0.3)
s2_gurultulu = gurultu_ekle(s2_temiz, 0.3)

# Ayri ayri DFT analizleri
dft_s1 = manuel_dft_1d(s1_gurultulu)
dft_s2 = manuel_dft_1d(s2_gurultulu)

# Iki sinyali topla ve toplam sinyalin DFT analizini yap
toplam_temiz = s1_temiz + s2_temiz
toplam_gurultulu = s1_gurultulu + s2_gurultulu
dft_toplam = manuel_dft_1d(toplam_gurultulu)

# 6. Gorev Testi: Frekans filtresi ile gurultuden arindirma
s1_filtrelenmis = frekans_filtresi(s1_gurultulu, esik_orani=0.3)
s2_filtrelenmis = frekans_filtresi(s2_gurultulu, esik_orani=0.3)
toplam_filtrelenmis = frekans_filtresi(toplam_gurultulu, esik_orani=0.25)

# Gorsellestirme
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.stem(np.abs(dft_s1[:N // 2]), linefmt="b-", markerfmt="bo", basefmt=" ", label="Sinyal 1 (f=50, A=12)")
plt.stem(np.abs(dft_s2[:N // 2]), linefmt="g-", markerfmt="go", basefmt=" ", label="Sinyal 2 (f=30, A=5)")
plt.title("5. Gorev: Ayri Ayri Sinyallerin DFT Harmonik Vektorleri")
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(toplam_gurultulu[:80], color="gray", label="Gurultulu Toplam Sinyal")
plt.plot(toplam_temiz[:80], color="blue", linestyle="--", label="Orijinal Temiz Toplam")
plt.title("5. Gorev: Toplam Sinyal (Ilk 80 Ornek)")
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.stem(np.abs(dft_toplam[:N // 2]), linefmt="r-", markerfmt="ro", basefmt=" ")
plt.title("5. Gorev: Toplam Sinyalin DFT Harmonik Vektoru (f=30 ve f=50 tepeleri)")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(toplam_temiz[:80], color="blue", linewidth=2, label="Orijinal Temiz Toplam")
plt.plot(toplam_filtrelenmis[:80], color="green", linestyle="--", linewidth=2, label="6. Gorev: Frekans Filtresi Ciktisi")
plt.title("6. Gorev: Esik Degerli Frekans Filtresi ile Yeniden Uretilen Sinyal")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cikti_task5_6_frekans_filtresi.png")
print("Task 5 ve 6 bitti.")