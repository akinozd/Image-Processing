import numpy as np
import matplotlib.pyplot as plt

# 1. Gorev: Boyutu 2'nin kuvveti olan vektoru sinus degerleri ile dolduran fonksiyon
def sinus_olustur(vektor, genlik, frekans, faz):
    N = len(vektor)  # Ornekleme sayisi (2'nin kuvveti)
    t = np.arange(N) / N
    for i in range(N):
        vektor[i] = genlik * np.sin(2 * np.pi * frekans * t[i] + faz)
    return vektor

# 2. Gorev: Genlik ve gurultu oranina [0, 1] gore rastgele gurultu ekleyen fonksiyon
def gurultu_ekle(vektor, gurultu_orani):
    genlik = np.max(np.abs(vektor))
    gurultulu = np.zeros(len(vektor))
    for i in range(len(vektor)):
        rastgele_gurultu = np.random.uniform(-1, 1) * genlik * gurultu_orani
        gurultulu[i] = vektor[i] + rastgele_gurultu
    return gurultulu

# Boyutu 2'nin kuvveti (512) olan bos vektor
N = 512
bos_vektor = np.zeros(N)

temiz_sinyal = sinus_olustur(bos_vektor, genlik=10, frekans=5, faz=0)
gurultulu_sinyal = gurultu_ekle(temiz_sinyal, gurultu_orani=0.3)

# Gorsellestirme ve kaydetme
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(temiz_sinyal, color="blue")
plt.title("1. Gorev: Temiz Sinuzoidal Dalga (N=512)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(gurultulu_sinyal, color="red")
plt.title("2. Gorev: Gurultu Eklenmis Sinyal (Oran = 0.3)")
plt.grid(True)

plt.tight_layout()
plt.savefig("cikti_task1_2_sinus_gurultu.png")
print("Task 1 ve 2 bitti.")