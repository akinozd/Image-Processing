import numpy as np
import matplotlib.pyplot as plt

# 1 ve 2. gorevdeki sinyal uretme ve gurultu ekleme fonksiyonlari
def sinus_olustur(vektor, genlik, frekans, faz):
    N = len(vektor)
    t = np.arange(N) / N
    for i in range(N):
        vektor[i] = genlik * np.sin(2 * np.pi * frekans * t[i] + faz)
    return vektor

def gurultu_ekle(vektor, gurultu_orani):
    genlik = np.max(np.abs(vektor))
    gurultulu = np.zeros(len(vektor))
    for i in range(len(vektor)):
        gurultulu[i] = vektor[i] + np.random.uniform(-1, 1) * genlik * gurultu_orani
    return gurultulu

# 3. Gorev: 1D Gauss ve 1D Medyan filtre fonksiyonlari
def gauss_filtre_1d(sinyal):
    # 5 elemanli basit Gauss cekirdegi
    cekirdek = np.array([1, 4, 6, 4, 1], dtype=float)
    cekirdek = cekirdek / np.sum(cekirdek)
    return np.convolve(sinyal, cekirdek, mode="same")

def medyan_filtre_1d(sinyal, pencere=5):
    N = len(sinyal)
    yarim = pencere // 2
    sonuc = np.zeros(N)
    for i in range(N):
        bas = max(0, i - yarim)
        bit = min(N, i + yarim + 1)
        sonuc[i] = np.median(sinyal[bas:bit])
    return sonuc

# 4. Gorev: Manuel 1D DFT fonksiyonu (harmonik vektorunu dondurur)
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

# Sinyalleri olustur (N = 256 -> 2'nin kuvveti)
N = 256
temiz = sinus_olustur(np.zeros(N), genlik=10, frekans=5, faz=0)
gurultulu = gurultu_ekle(temiz, gurultu_orani=0.3)

# Sirasiyla Gauss ve Medyan filtre uygula
gauss_sonrasi = gauss_filtre_1d(gurultulu)
medyan_sonrasi = medyan_filtre_1d(gauss_sonrasi, pencere=5)

# DFT harmonik vektorlerini hesapla
dft_temiz = manuel_dft_1d(temiz)
dft_gurultulu = manuel_dft_1d(gurultulu)

# Gorsellestirme
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(gurultulu, color="lightgray", label="Gurultulu")
plt.plot(gauss_sonrasi, color="orange", label="1. Gauss Filtre")
plt.plot(medyan_sonrasi, color="green", label="2. Gauss + Medyan Filtre")
plt.title("3. Gorev: Sirasiyla Gauss ve Medyan Filtre Sonuclari")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(np.abs(dft_temiz[:N // 2]), linefmt="b-", markerfmt="bo", basefmt=" ")
plt.title("4. Gorev: Gurultusuz Sinyal DFT Harmonik Vektoru (Sadece f=5 tepe noktasi var)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(np.abs(dft_gurultulu[:N // 2]), linefmt="r-", markerfmt="ro", basefmt=" ")
plt.title("4. Gorev: Gurultulu Sinyal DFT Harmonik Vektoru (Tum frekanslara gurultu yayilmis)")
plt.grid(True)

plt.tight_layout()
plt.savefig("cikti_task3_4_filtre_dft.png")

print("Task 3 ve 4 bitti.")
print("Yorum (Task 3): Gauss filtresi yuksek frekansli titremeleri yumusatir, ardindan gelen Medyan filtresi ani sivri sicramalari bastirarak sinyali ana sinus formuna yaklastirir.")
print("Yorum (Task 4): Gurultusuz sinyalin DFT'sinde sadece ana frekansta (k=5) keskin bir tepe gorulurken, gurultulu sinyalde tum frekans bilesenlerine yayilmis kucuk harmonikler olusur.")