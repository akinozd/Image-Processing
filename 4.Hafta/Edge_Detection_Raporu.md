# Kenar Tespit Algoritmaları Karşılaştırma Raporu (4. Hafta)

Bu çalışmada Prewitt, Sobel, Laplacian, Laplacian of Gaussian (LoG) ve Canny filtrelerinin kenar tespiti üzerindeki davranışları tuğla duvar görseli üzerinde test edilmiş ve analiz edilmiştir.

---

### 1. Prewitt ve Sobel Operatörlerinin Karşılaştırması (Task 1, 2, 3, 4)
* **Matematiksel Yapı:** Her iki filtre de 1. dereceden türev (gradyan) operatörüdür. Prewitt homojen katsayılara (`[-1, 0, 1]`) sahipken; Sobel merkez piksele 2 ağırlığını vererek (`[-1, 0, 1]`, `[-2, 0, 2]`) küçük bir Gauss yumuşatması uygular.
* **Sonuç Analizi:** Yatay filtreler duvardaki uzun harç hatlarını, dikey filtreler ise şaşırtmalı dikey derzleri öne çıkarmıştır. Sobel operatörü, merkez piksele verdiği ağırlık sayesinde Prewitt'e göre doku parazitlerini daha iyi bastırmış ve daha belirgin kenar hatları üretmiştir.

---

### 2. Birinci Derece (Sobel/Prewitt) vs İkinci Derece (Laplacian) (Task 5)
* **Yön Bağımsızlığı:** Sobel ve Prewitt yatay ve dikey yönde ayrı türevler alıp bileşke oluştururken; Laplacian 2. derece türev operatörüdür ve yön bağımsızdır (izotropik). Tüm yönlerdeki parlaklık değişimlerini tek bir konvolüsyonla yakalar.
* **Gürültü Duyarlılığı:** 2. türev gürültüye karşı çok hassastır. Öncesinde Gauss yumuşatması uygulanmadığında tuğla yüzeyindeki mikro pürüzleri kenar gibi algılar. Bu nedenle Gauss ön filtresi kritik öneme sahiptir.

---

### 3. Laplacian of Gaussian (LoG - 7x7) Analizi (Task 6)
* LoG, Gauss yumuşatma ile Laplacian türevini $7 \times 7$ boyutundaki tek bir çekirdekte birleştirir.
* $3 \times 3$ standart Laplacian'a göre daha geniş bir etki alanına (receptive field) sahiptir. Bu sayede görseldeki küçük dokusal kumlanmaları bastırarak tuğlaların gerçek sınır hatlarını sıfır geçişleri (zero-crossing) mantığıyla daha pürüzsüz tespit etmiştir.

---

### 4. Canny Kenar Algoritması ve Performans Kıyaslaması (Task 7)
* **Kenar İnceltme ve Doğruluk:** Canny sadece basit bir konvolüsyon filtresi değil, çok adımlı bir boru hattıdır (Gauss -> Gradyan -> Non-Maximum Suppression -> Histerezis Eşikleme). Çıktısı diğer filtreler gibi gri tonlu gradyan haritası değil, **tek piksel kalınlığında ikili (binary)** kenarlardır.
* **Çalışma Zamanı Kıyaslaması:** Basit türev filtrelerine (Sobel/Prewitt) göre arka planda histerezis ve lokal tepe noktası araması yaptığı için teorik işlem yükü daha fazladır; ancak OpenCV'nin C++ optimizasyonu sayesinde milisaniyeler seviyesinde çalışır ve en temiz kenar haritasını üretir.