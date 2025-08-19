AutoHomographyMatcher — Otomatik Homografi ile İki Kamerayı Hizalama & Panorama
Bu depo, ortak alanı gören iki kamera/videodan otomatik olarak homografi matrisi hesaplayıp görüntüleri aynı düzlemde hizalayan ve panorama üreten bir Python sınıfı (AutoHomographyMatcher) içerir. Manuel referans noktası (corner/correspondence) seçimi gerektirmez; SIFT/ORB + FLANN/BF eşleştiricisi ve RANSAC ile güvenilir eşleşmeleri otomatik seçer.
✨ Özellikler
Sıfır manuel işaretleme: Özellik çıkarım + eşleştirme + RANSAC tümü otomatik.
Seçilebilir dedektör ve eşleştirici:
Dedektör: SIFT veya ORB
Eşleştirici: FLANN veya BF (Brute Force)
Esnek görselleştirme: Tüm eşleşmeler ve RANSAC sonrası iyi eşleşmelerin çizimi.
Panorama üretimi: İki görüntüyü tek düzlemede birleştirir.
Video veya görüntü ile çalışma: Videodan belirli frame’i çeker veya doğrudan resim dosyalarını kullanır.
Çıktıları kaydetme: Panorama JPG ve homografi matrisini NPY olarak kaydeder.
Özelleştirilebilir eşik: min_matches, nfeatures, oran testi eşiği (0.8) vb.
📦 Gereksinimler
Python 3.8+
Paketler:
opencv-contrib-python (SIFT için contrib sürümü gerekir)
numpy
matplotlib
pip install opencv-contrib-python numpy matplotlib
Not: opencv-python yerine opencv-contrib-python kurulu olduğundan emin olun; aksi halde SIFT bulunamaz.
📁 Proje Dosyaları
your_script.py (aşağıdaki sınıf ve main() fonksiyonu burada)
AutoHomographyMatcher sınıfı
main() — iki video ile uçtan uca örnek
test_with_images() — iki resimle hızlı test
⚙️ Nasıl Çalışır?
Frame çıkarımı: Her videodan seçilen frame okunur.
Özellik çıkarımı: SIFT/ORB ile anahtar noktalar + tanımlayıcılar (detectAndCompute).
Eşleştirme: FLANN/BF ile knnMatch(k=2) ve Lowe’s ratio test (0.8) uygulanır.
RANSAC homografisi: Gürültülü eşleşmeler elenir, 3×3 homografi elde edilir.
Warp & birleştirme: Homografi ile görüntüler aynı düzleme aktarılır ve panorama oluşturulur.
Görselleştirme & kayıt: Sonuçlar çizilir ve dosyaya yazılır.
🚀 Hızlı Başlangıç (Video ile)
Script’in sonundaki main() fonksiyonundaki yolları düzenleyin:
video1_path = 'ShopAssistant1cor.mpg'   # Kamera 1 (A+B bölgesi)
video2_path = 'ShopAssistant1front.mpg' # Kamera 2 (B+C bölgesi)
Çalıştırın:
python your_script.py
Beklenen çıktılar:
Eşleşme sayıları ve homografi matrisi terminalde
panorama_result.jpg
homography_matrix.npy
🖼️ Hızlı Test (Resim ile)
from your_script import test_with_images

H, pano = test_with_images('img1.jpg', 'img2.jpg')
H: 3×3 homografi
pano: panorama görüntüsü (NumPy array)
🛠️ API (Sınıf) Referansı
AutoHomographyMatcher(detector_type='SIFT', matcher_type='FLANN', min_matches=50)
detector_type: 'SIFT' veya 'ORB'
matcher_type: 'FLANN' veya 'BF'
min_matches: RANSAC öncesi iyi eşleşmelerin alt limiti
extract_frame(video_path, frame_number=0) -> np.ndarray
Belirli bir frame’i okur (BGR).
detect_and_compute(image) -> (keypoints, descriptors)
Girdi: BGR veya gri görüntü
Çıkış: anahtar noktalar + tanımlayıcılar
match_features(desc1, desc2) -> List[DMatch]
knnMatch(k=2) ve oran testi (0.8) ile iyi eşleşmeleri döndürür.
find_homography_ransac(kp1, kp2, matches) -> (H, mask, good_matches)
cv2.findHomography(..., method=cv2.RANSAC, ransacReprojThreshold=5.0, confidence=0.99, maxIters=5000)
H: 3×3 homografi matrisi
visualize_matches(img1, kp1, img2, kp2, matches, title="...") -> np.ndarray
Eşleşmeleri tek bir canvas üzerinde çizer.
create_panorama(img1, img2, H) -> (panorama, img1_warped, img2_warped)
İkinci görüntüyü birincinin düzlemine warplar, basit birleştirme uygular.
process_videos(video1_path, video2_path, frame1=0, frame2=0, visualize=True) -> dict
Uçtan uca akış. Dönen sözlük:
{
  'homography_3x3': H,
  'homography_4x4': H4,        # 3×3 H’nin 4×4’e gömümü (projeksiyon/3D entegrasyon için)
  'panorama': pano,
  'img1_warped': img1w,
  'img2_warped': img2w,
  'matches_count': int,
  'good_matches_count': int,
  'keypoints_img1': int,
  'keypoints_img2': int
}
🔧 Parametreler ve İnce Ayar
Dedektör seçimi
SIFT: Daha sağlam, genelde daha doğru, biraz daha yavaş.
ORB: Hızlı, ikili tanımlayıcı; ışık/ölçek değişiminde SIFT kadar sağlam olmayabilir.
Eşleştirici seçimi
FLANN: Büyük veri (çok sayıda descriptor) için hızlı.
BF: Basit ve güvenilir; küçük/orta boyutlarda yeterli.
min_matches
Ortak alan dar ise veya sahne tekrarlı desen içeriyorsa düşürün (örn. 15–30).
Oran testi eşiği (0.8)
Daha sıkı (örn. 0.7): Yanlış eşleşmeler azalır, ama eşleşme sayısı düşebilir.
Daha gevşek (örn. 0.85): Eşleşme sayısı artar, RANSAC daha fazla eleme yapar.
📊 İyi Sonuçlar İçin İpuçları
Ortak alan (overlap) görünür olmalı; hareket bulanıklığı az, tekstürlü bölgeler tercih edilir.
Çözünürlük: Çok büyük kareler yerine, gerekirse makul ölçeğe küçültün (performans artar).
Dedektör nfeatures: SIFT_create(nfeatures=10000) ve ORB_create(nfeatures=10000) ayarlanabilir.
Farklı frame’ler deneyin: Özellikle iki videodaki senkron farklıysa.
Aydınlatma/kontrast: Kodda gri görüntüye histogram eşitleme uygulanır; düşük kontrastı toparlar.
🧪 Örnek Kullanımlar
1) SIFT + FLANN ile
matcher = AutoHomographyMatcher(detector_type='SIFT', matcher_type='FLANN', min_matches=30)
res = matcher.process_videos('cam1.mp4', 'cam2.mp4', frame1=100, frame2=120, visualize=True)
2) ORB + BF ile (daha hızlı)
matcher = AutoHomographyMatcher(detector_type='ORB', matcher_type='BF', min_matches=20)
Hres = matcher.process_videos('cam1.mp4', 'cam2.mp4', frame1=0, frame2=0, visualize=False)
3) İki resimle hızlı test
H, pano = test_with_images('kafe_cam1.jpg', 'kafe_cam2.jpg')
🧯 Hata Giderme (Troubleshooting)
Video açılamadı: ...
Yol doğru mu? Dosya uzantısı/izinler? Codec desteği?
Frame okunamadı
frame_number aralık dışında olabilir. Daha küçük bir frame numarası deneyin.
Yeterli anahtar nokta bulunamadı!
Görüntü çok düz/tekstürsüz olabilir; farklı frame veya ORB deneyin, çözünürlüğü büyütün.
Homografi hesaplanamadı
min_matches değerini düşürün (örn. 15).
Oran test eşiğini 0.8 → 0.85 yapmayı deneyin.
Daha yüksek özellik sayısı (nfeatures) veya farklı eşleştirici deneyin.
Panorama’da kırpma veya kayma
Ortak alan az veya parallax büyük olabilir (kamera merkezleri arası fark). Daha iyi overlap içeren frame’ler seçin.
🧩 Notlar
4×4 homografi (homography_4x4), 3×3 projektif dönüşümün 4×4 matrise gömülmüş halidir. 3D/AR pipeline’larında veya shader/matris zincirlerinde kullanım kolaylığı için eklenmiştir; 3D dış dünya anlamına gelmez.
Kod, basit maksimum alma ile birleştirir (np.maximum). Daha kaliteli blend için çok bantlı birleştirme (multi-band blending) eklenebilir.
📜 Lisans
İstediğiniz lisansı ekleyin (örneğin MIT). Örnek:
MIT License
Copyright (c) 2025 ...
🤝 Katkı
İyileştirmeler (ör. çok bantlı blend, akıllı maskeleme, otomatik overlap kestirimi) için PR’lar memnuniyetle karşılanır.
Test videoları/örnek veri paylaşımı sonuç kalitesini artırır.
🗺️ Yol Haritası (Opsiyonel)
 Çok bantlı panorama birleştirme
 Otomatik overlap bölgesi kestirimi (grid tabanlı skor)
 Hareket telafisi (optik akış) ile “daha eş-anda” frame seçimi
 CLI arayüz: --detector, --matcher, --min-matches, --frame1, --frame2 bayrakları
🔗 Hızlı Özet Kod Parçası
matcher = AutoHomographyMatcher(detector_type='SIFT', matcher_type='FLANN', min_matches=15)
results = matcher.process_videos('ShopAssistant1cor.mpg', 'ShopAssistant1front.mpg',
                                 frame1=0, frame2=0, visualize=True)
cv2.imwrite('panorama_result.jpg', results['panorama'])
np.save('homography_matrix.npy', results['homography_3x3'])
