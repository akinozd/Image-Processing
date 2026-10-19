5. Hafta: Renk Uzayları ve X-Ray Sahte Renklendirme (Pseudocoloring) Raporu

Bu çalışmada temel renk uzayı dönüşümleri incelenmiş, Kaggle `HUMS-X-Ray-Dataset` altındaki `ThreadsRGB` görüntüleri gri seviyeye dönüştürülüp güvenlik operatörleri için sahte renklendirme (pseudocoloring) teknikleriyle işlenmiştir.

---

 1. Renk Paleti Seçimi ve Gerekçesi
* İnsan Algısı:** İnsan gözü tek kanallı gri tonlamalı bir görüntüde yalnızca 30-40 farklı gri parlaklık basamağını ayırt edebilirken, renkli spektrumda binlerce farklı tonu algılayabilir.
* COLORMAP_TURBO / JET Tercihi: Bagaj ve kargo taramalarında operatörün uzun süreli ekran takibinde dikkat dağınıklığını önlemek ve şüpheli nesneleri (bıçak, silah, kablo vb.) anında fark edebilmesini sağlamak için algısal olarak lineer ve kontrastı yüksek olan TURBO renk paleti seçilmiştir. 
* Bu palet, X-ışını absorpsiyonu yüksek olan yoğun metalik nesneleri zıt renk tonlarında (mavi/açık yeşil), geçirgenliği yüksek boşluk/arka plan alanlarını ise koyu kırmızı/sıcak tonlarda toplayarak nesne sınırlarını belirginleştirir.

---

2. Üretilen Sahte Renkli Görüntü ile Orijinal Görüntü Arasındaki Farklar
* Fiziksel Ayrım (Orijinal Görüntü): Orijinal `ThreadsRGB` görüntüleri, Çift Enerjili X-Işını (Dual-Energy X-Ray) sistemleriyle elde edilmiştir. Bu sistemler malzemelerin efektif atom numarasına ($Z_{eff}$) bakarak kimyasal ayrım yapar:
  * Organik maddeler (patlayıcı, uyuşturucu, plastik, kumaş - düşük Z) Turuncu
  * İnorganik maddeler (cam, alüminyum, kemik - orta Z) Yeşil
  * Ağır metaller (silah, bıçak, kurşun - yüksek Z) Mavi
* Yoğunluk Temelli Ayrım (Bizim Çıktımız): Bizim ürettiğimiz görüntü ise tek kanallı gri seviyenin piksel yoğunluğuna (absorpsiyon miktarına) uygulanan matematiksel bir **Lookup Table (LUT)** eşlemesidir. 
* Yorum: Sahte renklendirme kimyasal malzeme türünü (organik/inorganik) orijinal donanım gibi ayıramasa da, nesnelerin geometrik sınırlarını, üst üste binmiş kenarları ve ince metalik tehditleri gri seviyeye kıyasla güvenlik operatörü için belirgin hale getirmiştir.

---

3. CUDA ile Paralel Hızlandırma Mimarisi
* Hafıza Transferi (Host to Device): Görüntü pikselleri CPU RAM'inden (Host) GPU Global Memory'ye (Device) kopyalanmıştır.
* İş Parçacığı (Thread) Mantığı: CPU üzerinde pikseller iç içe iki `for` döngüsüyle sırayla gezilirken ($O(W \times H)$), CUDA mimarisinde her bir piksel veya kanal bağımsız bir iş parçacığına (thread) atanmış ve $O(1)$ paralel zamanda işlenmiştir.
* Histogram ve Eş Zamanlılık: Histogram hesaplamasında birden fazla GPU çekirdeğinin aynı histogram kutusuna (bin) aynı anda yazmasını engellemek için atomik işlemler (`atomicAdd`) ve `bincount` paralelleştirmesi kullanılmıştır.