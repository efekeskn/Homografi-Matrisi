# AutoHomographyMatcher — Otomatik Homografi ile İki Kamerayı Hizalama & Panorama

**AutoHomographyMatcher**, ortak alanı gören iki kameradan (veya iki görüntüden) **manuel işaretleme gerektirmeden** homografi matrisi hesaplar, görüntüleri **aynı düzleme hizalar** ve **panorama** üretir.  
SIFT/ORB + FLANN/BF eşleştirme ve **RANSAC** ile güvenilir eşleşmeler otomatik seçilir.

---

## İçindekiler
- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç (Video)](#hızlı-başlangıç-video)
- [Hızlı Test (Görüntü)](#hızlı-test-görüntü)
- [API Referansı](#api-referansı)
- [Parametreler ve İnce Ayar](#parametreler-ve-ince-ayar)
- [İpuçları](#i̇puçları)
- [Hata Giderme](#hata-giderme)
- [Notlar](#notlar)
- [Yol Haritası](#yol-haritası)
- [Lisans](#lisans)
- [Hızlı Özet Kod Parçası](#hızlı-özet-kod-parçası)

---

## Özellikler
- **Sıfır manuel işaretleme:** Özellik çıkarım + eşleştirme + RANSAC tamamen otomatik.
- **Seçilebilir dedektör & eşleştirici:**  
  - Dedektör: `SIFT` veya `ORB`  
  - Eşleştirici: `FLANN` veya `BF (Brute Force)`
- **Görselleştirme:** Tüm eşleşmeler ve RANSAC sonrası iyi eşleşmelerin çizimi.
- **Panorama üretimi:** İki görüntü tek düzlemde birleştirilir.
- **Video veya görüntü desteği:** Videodan frame çıkarır veya doğrudan resim dosyalarıyla çalışır.
- **Çıktıları kaydetme:** Panorama `.jpg`, homografi `.npy`.

---

## Gereksinimler
- Python 3.8+
- Paketler:
  - `opencv-contrib-python` (**SIFT** için *contrib* gerekir)
  - `numpy`
  - `matplotlib`

### Kurulum
```bash
pip install opencv-contrib-python numpy matplotlib

