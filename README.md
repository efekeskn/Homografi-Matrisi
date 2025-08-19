# Ortak Alan Tespiti - Homografi Matrisi

Bu proje, iki farklı kamera açısından çekilmiş video görüntüleri arasında ortak alanları tespit ederek homografi matrisi hesaplayan bir bilgisayarlı görü projesidir. Proje, SIFT ve ORB gibi özellik tespit algoritmaları kullanarak görüntüler arasında eşleşme bulur ve bu eşleşmeleri kullanarak iki görüntüyü birleştirip panorama oluşturur.

## 🎯 Proje Amacı

- İki farklı kamera açısından çekilmiş video görüntüleri arasında ortak alanları tespit etmek
- Homografi matrisi hesaplayarak görüntüleri hizalamak
- Birleştirilmiş panorama görüntüsü oluşturmak
- Farklı özellik tespit ve eşleştirme algoritmalarını karşılaştırmak

## ✨ Özellikler

- **Sıfır Manuel İşaretleme**: Özellik çıkarım + eşleştirme + RANSAC tümü otomatik
- **Çoklu Detector Desteği**: SIFT ve ORB algoritmaları
- **Esnek Matcher Seçenekleri**: FLANN ve Brute Force eşleştirme
- **RANSAC Filtreleme**: Güvenilir eşleşmeleri filtrelemek için
- **Görselleştirme**: Eşleşmeleri ve sonuçları görsel olarak gösterme
- **Video Frame Çıkarma**: Video dosyalarından belirli frame'leri çıkarma
- **Panorama Oluşturma**: İki görüntüyü birleştirerek panorama oluşturma
- **Çıktıları Kaydetme**: Panorama JPG ve homografi matrisini NPY olarak kaydeder
- **Özelleştirilebilir Eşik**: min_matches, nfeatures, oran testi eşiği (0.8) vb.

## 📋 Gereksinimler

```bash
pip install opencv-contrib-python  # SIFT için contrib sürümü gerekir
pip install numpy
pip install matplotlib
```

**Not**: `opencv-python` yerine `opencv-contrib-python` kurulu olduğundan emin olun; aksi halde SIFT bulunamaz.

## 🚀 Kullanım

### Temel Kullanım

```python
from homografi_detector import AutoHomographyMatcher

# Homografi matcher oluştur
matcher = AutoHomographyMatcher(
    detector_type='SIFT',    # 'SIFT' veya 'ORB'
    matcher_type='FLANN',    # 'FLANN' veya 'BF'
    min_matches=15           # Minimum gerekli eşleşme sayısı
)

# Video dosyalarını işle
results = matcher.process_videos(
    video1_path='video1.mpg',
    video2_path='video2.mpg',
    frame1=0,       # İlk videodan frame numarası
    frame2=0,       # İkinci videodan frame numarası
    visualize=True  # Görselleştirme açık/kapalı
)
```

### Hızlı Test (Resim ile)

```python
from homografi_detector import test_with_images

H, pano = test_with_images('img1.jpg', 'img2.jpg')
# H: 3×3 homografi
# pano: panorama görüntüsü (NumPy array)
```

### Parametreler

- **detector_type**: Özellik tespit algoritması ('SIFT' veya 'ORB')
- **matcher_type**: Eşleştirme algoritması ('FLANN' veya 'BF')
- **min_matches**: Homografi hesaplamak için gereken minimum eşleşme sayısı

### Çıktılar

- **homography_3x3**: 3x3 homografi matrisi
- **homography_4x4**: 4x4 homografi matrisi (3×3 H'nin 4×4'e gömümü)
- **panorama**: Birleştirilmiş panorama görüntüsü
- **matches_count**: Toplam eşleşme sayısı
- **good_matches_count**: RANSAC sonrası güvenilir eşleşme sayısı

## 📁 Dosya Yapısı

```
Ortak Alan Tespiti/
├── homografi_detector.py      # Ana kod dosyası
├── homography_matrix.npy      # Hesaplanan homografi matrisi
├── panorama_result.jpg        # Oluşturulan panorama
├── README.md                  # Bu dosya
├── ShopAssistant1cor.mpg      # Örnek video 1
└── ShopAssistant1front.mpg    # Örnek video 2
```

## ⚙️ Nasıl Çalışır?

1. **Frame çıkarımı**: Her videodan seçilen frame okunur
2. **Özellik çıkarımı**: SIFT/ORB ile anahtar noktalar + tanımlayıcılar (detectAndCompute)
3. **Eşleştirme**: FLANN/BF ile knnMatch(k=2) ve Lowe's ratio test (0.8) uygulanır
4. **RANSAC homografisi**: Gürültülü eşleşmeler elenir, 3×3 homografi elde edilir
5. **Warp & birleştirme**: Homografi ile görüntüler aynı düzleme aktarılır ve panorama oluşturulur
6. **Görselleştirme & kayıt**: Sonuçlar çizilir ve dosyaya yazılır

## 🔧 Sınıf Metodları

### AutoHomographyMatcher

#### `__init__(detector_type, matcher_type, min_matches)`
Homografi matcher'ı başlatır.

#### `extract_frame(video_path, frame_number)`
Video dosyasından belirli bir frame'i çıkarır (BGR formatında).

#### `detect_and_compute(image)`
Görüntüde anahtar noktaları tespit eder ve tanımlayıcıları hesaplar.

#### `match_features(desc1, desc2)`
İki görüntünün tanımlayıcılarını eşleştirir (knnMatch(k=2) ve oran testi ile).

#### `find_homography_ransac(kp1, kp2, matches)`
RANSAC algoritması ile güvenilir homografi matrisi hesaplar.

#### `create_panorama(img1, img2, homography)`
İki görüntüyü homografi matrisi ile birleştirip panorama oluşturur.

#### `process_videos(video1_path, video2_path, frame1, frame2, visualize)`
Ana işlem fonksiyonu - videoları işleyerek homografi hesaplar.

## 🔧 Parametreler ve İnce Ayar

### Dedektör Seçimi
- **SIFT**: Daha sağlam, genelde daha doğru, biraz daha yavaş
- **ORB**: Hızlı, ikili tanımlayıcı; ışık/ölçek değişiminde SIFT kadar sağlam olmayabilir

### Eşleştirici Seçimi
- **FLANN**: Büyük veri (çok sayıda descriptor) için hızlı
- **BF**: Basit ve güvenilir; küçük/orta boyutlarda yeterli

### min_matches
Ortak alan dar ise veya sahne tekrarlı desen içeriyorsa düşürün (örn. 15–30).

### Oran Testi Eşiği (0.8)
- Daha sıkı (örn. 0.7): Yanlış eşleşmeler azalır, ama eşleşme sayısı düşebilir
- Daha gevşek (örn. 0.85): Eşleşme sayısı artar, RANSAC daha fazla eleme yapar

## 📊 İyi Sonuçlar İçin İpuçları

- Ortak alan (overlap) görünür olmalı; hareket bulanıklığı az, tekstürlü bölgeler tercih edilir
- Çözünürlük: Çok büyük kareler yerine, gerekirse makul ölçeğe küçültün (performans artar)
- Dedektör nfeatures: SIFT_create(nfeatures=10000) ve ORB_create(nfeatures=10000) ayarlanabilir
- Farklı frame'ler deneyin: Özellikle iki videodaki senkron farklıysa
- Aydınlatma/kontrast: Kodda gri görüntüye histogram eşitleme uygulanır; düşük kontrastı toparlar

## 🧪 Örnek Kullanımlar

### 1) SIFT + FLANN ile
```python
matcher = AutoHomographyMatcher(detector_type='SIFT', matcher_type='FLANN', min_matches=30)
res = matcher.process_videos('cam1.mp4', 'cam2.mp4', frame1=100, frame2=120, visualize=True)
```

### 2) ORB + BF ile (daha hızlı)
```python
matcher = AutoHomographyMatcher(detector_type='ORB', matcher_type='BF', min_matches=20)
Hres = matcher.process_videos('cam1.mp4', 'cam2.mp4', frame1=0, frame2=0, visualize=False)
```

### 3) İki resimle hızlı test
```python
H, pano = test_with_images('kafe_cam1.jpg', 'kafe_cam2.jpg')
```

## 🛠️ Sorun Giderme

### Yaygın Hatalar ve Çözümleri

1. **"Yeterli anahtar nokta bulunamadı"**
   - Farklı frame numaraları deneyin
   - Detector tipini değiştirin (SIFT ↔ ORB)
   - Görüntü çok düz/tekstürsüz olabilir; çözünürlüğü büyütün

2. **"Homografi hesaplanamadı"**
   - `min_matches` değerini düşürün (örn. 15)
   - Videoların ortak alanı olduğundan emin olun
   - Farklı matcher tipi deneyin
   - Oran test eşiğini 0.8 → 0.85 yapmayı deneyin

3. **Video açılamadı hatası**
   - Video dosya yollarını kontrol edin
   - Dosya formatının desteklendiğinden emin olun
   - Codec desteğini kontrol edin

4. **Frame okunamadı**
   - frame_number aralık dışında olabilir. Daha küçük bir frame numarası deneyin

5. **Panorama'da kırpma veya kayma**
   - Ortak alan az veya parallax büyük olabilir (kamera merkezleri arası fark)
   - Daha iyi overlap içeren frame'ler seçin

## 📝 Notlar

- SIFT algoritması daha doğru sonuçlar verir ancak daha yavaştır
- ORB algoritması daha hızlıdır ancak daha az doğru olabilir
- FLANN matcher genellikle daha hızlıdır
- RANSAC parametreleri görüntü kalitesine göre ayarlanabilir
- 4×4 homografi (homography_4x4), 3×3 projektif dönüşümün 4×4 matrise gömülmüş halidir. 3D/AR pipeline'larında veya shader/matris zincirlerinde kullanım kolaylığı için eklenmiştir
- Kod, basit maksimum alma ile birleştirir (np.maximum). Daha kaliteli blend için çok bantlı birleştirme (multi-band blending) eklenebilir

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

İyileştirmeler (ör. çok bantlı blend, akıllı maskeleme, otomatik overlap kestirimi) için PR'lar memnuniyetle karşılanır.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Geliştirici

Bu proje bilgisayarlı görü ve homografi matrisi hesaplama konularında eğitim amaçlı geliştirilmiştir.

## 🗺️ Yol Haritası (Opsiyonel)

- Çok bantlı panorama birleştirme
- Otomatik overlap bölgesi kestirimi (grid tabanlı skor)
- Hareket telafisi (optik akış) ile "daha eş-anda" frame seçimi
- CLI arayüz: --detector, --matcher, --min-matches, --frame1, --frame2 bayrakları

## 🔗 Hızlı Özet Kod Parçası

```python
matcher = AutoHomographyMatcher(detector_type='SIFT', matcher_type='FLANN', min_matches=15)
results = matcher.process_videos('ShopAssistant1cor.mpg', 'ShopAssistant1front.mpg',
                                 frame1=0, frame2=0, visualize=True)
cv2.imwrite('panorama_result.jpg', results['panorama'])
np.save('homography_matrix.npy', results['homography_3x3'])
```
