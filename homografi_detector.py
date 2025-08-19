import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AutoHomographyMatcher:
    def __init__(self, detector_type='SIFT', matcher_type='FLANN', min_matches=50):
        """
        Otomatik homografi hesaplama ve görüntü hizalama sınıfı
        
        Args:
            detector_type: 'SIFT' veya 'ORB'
            matcher_type: 'FLANN' veya 'BF' (Brute Force)
            min_matches: Minimum gerekli eşleşme sayısı
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.min_matches = min_matches
        
        # Detector seçimi
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=10000)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=10000)
        else:
            raise ValueError("Detector tipi 'SIFT' veya 'ORB' olmalı")
        
        # Matcher seçimi
        if matcher_type == 'FLANN':
            if detector_type == 'SIFT':
                # SIFT için FLANN parametreleri
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:
                # ORB için FLANN parametreleri
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'BF':
            if detector_type == 'SIFT':
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError("Matcher tipi 'FLANN' veya 'BF' olmalı")
    
    def extract_frame(self, video_path, frame_number=0):
        """Video dosyasından belirli frame'i çıkart"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Frame okunamadı: {video_path}, frame: {frame_number}")
        
        return frame
    
    def detect_and_compute(self, image):
        """Görüntüde anahtar noktaları tespit et ve tanımlayıcıları hesapla"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Histogram eşitleme ile kontrast artırma
        gray = cv2.equalizeHist(gray)
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """İki görüntünün tanımlayıcılarını eşleştir"""
        if desc1 is None or desc2 is None:
            return []
        
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            if self.matcher_type == 'FLANN':
                # FLANN matcher için descriptor tipini kontrol et
                if self.detector_type == 'ORB':
                    desc1 = desc1.astype(np.uint8)
                    desc2 = desc2.astype(np.uint8)
                else:
                    desc1 = desc1.astype(np.float32)
                    desc2 = desc2.astype(np.float32)
                
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
            else:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Lowe's ratio test ile iyi eşleşmeleri filtrele (daha esnek)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:  # 0.7'den 0.8'e çıkarıldı
                        good_matches.append(m)
            
            return good_matches
            
        except Exception as e:
            print(f"Eşleştirme hatası: {e}")
            return []
    
    def find_homography_ransac(self, kp1, kp2, matches):
        """RANSAC ile güvenilir homografi matrisi hesapla"""
        if len(matches) < self.min_matches:
            return None, None, []
        
        # Eşleşen noktaları çıkart
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC ile homografi hesapla
        homography, mask = cv2.findHomography(
            dst_pts, src_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=5000
        )
        
        # İnlier eşleşmeleri filtrele
        if mask is not None:
            good_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        else:
            good_matches = []
        
        return homography, mask, good_matches
    
    def visualize_matches(self, img1, kp1, img2, kp2, matches, title="Eşleşmeler"):
        """Eşleşen noktaları görselleştir"""
        # Eşleşmeleri çiz
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Matplotlib ile göster
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'{title} (Toplam: {len(matches)} eşleşme)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return img_matches
    
    def create_panorama(self, img1, img2, homography):
        """İki görüntüyü homografi ile birleştirip panorama oluştur"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # İkinci görüntünün köşe noktalarını birinci görüntüye dönüştür
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners2, homography)
        
        # Tüm köşe noktalarını birleştir
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners1, transformed_corners), axis=0)
        
        # Sınırlayıcı kutu hesapla
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Kaydırma matrisi oluştur
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        # Panorama boyutunu hesapla
        panorama_width = x_max - x_min
        panorama_height = y_max - y_min
        
        # İkinci görüntüyü dönüştür ve kaydır
        img2_warped = cv2.warpPerspective(
            img2, 
            translation @ homography, 
            (panorama_width, panorama_height)
        )
        
        # Birinci görüntüyü kaydır
        img1_warped = cv2.warpPerspective(
            img1, 
            translation, 
            (panorama_width, panorama_height)
        )
        
        # Basit blending ile görüntüleri birleştir
        panorama = np.maximum(img1_warped, img2_warped)
        
        return panorama, img1_warped, img2_warped
    
    def process_videos(self, video1_path, video2_path, frame1=0, frame2=0, visualize=True):
        """İki video dosyasını işleyerek homografi hesapla"""
        
        print(f"Video 1'den frame {frame1} çıkartılıyor...")
        img1 = self.extract_frame(video1_path, frame1)
        
        print(f"Video 2'den frame {frame2} çıkartılıyor...")
        img2 = self.extract_frame(video2_path, frame2)
        
        print(f"Görüntüler yüklendi: {img1.shape}, {img2.shape}")
        
        # Anahtar noktaları tespit et
        print("Anahtar noktalar tespit ediliyor...")
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)
        
        print(f"Tespit edilen noktalar - Görüntü 1: {len(kp1)}, Görüntü 2: {len(kp2)}")
        
        if len(kp1) < 10 or len(kp2) < 10:
            raise ValueError("Yeterli anahtar nokta bulunamadı!")
        
        # Eşleştirme yap
        print("Özellik eşleştirme yapılıyor...")
        matches = self.match_features(desc1, desc2)
        print(f"İlk eşleştirmede {len(matches)} eşleşme bulundu")
        
        if visualize and len(matches) > 0:
            self.visualize_matches(img1, kp1, img2, kp2, matches, "Tüm Eşleşmeler")
        
        # Homografi hesapla
        print("RANSAC ile homografi hesaplanıyor...")
        homography, mask, good_matches = self.find_homography_ransac(kp1, kp2, matches)
        
        if homography is None:
            raise ValueError(f"Homografi hesaplanamadı! En az {self.min_matches} güvenilir eşleşme gerekli, {len(matches)} bulundu.")
        
        print(f"RANSAC sonrası {len(good_matches)} güvenilir eşleşme kaldı")
        
        if visualize and len(good_matches) > 0:
            self.visualize_matches(img1, kp1, img2, kp2, good_matches, "Güvenilir Eşleşmeler (RANSAC sonrası)")
        
        # Panorama oluştur
        print("Panorama oluşturuluyor...")
        panorama, img1_warped, img2_warped = self.create_panorama(img1, img2, homography)
        
        if visualize:
            # Sonuçları göster
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f'Kamera 1 (Frame {frame1})', fontsize=14)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Kamera 2 (Frame {frame2})', fontsize=14)
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(cv2.cvtColor(img2_warped, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Kamera 2 - Hizalanmış', fontsize=14)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('Birleştirilmiş Panorama', fontsize=14)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Homografi matrisini 4x4 formatına çevir
        homography_4x4 = np.eye(4, dtype=np.float32)
        homography_4x4[:3, :3] = homography
        
        results = {
            'homography_3x3': homography,
            'homography_4x4': homography_4x4,
            'panorama': panorama,
            'img1_warped': img1_warped,
            'img2_warped': img2_warped,
            'matches_count': len(matches),
            'good_matches_count': len(good_matches),
            'keypoints_img1': len(kp1),
            'keypoints_img2': len(kp2)
        }
        
        return results

# Kullanım örneği
def main():
    # Video dosya yolları (bu kısımları kendi dosyalarınızın yolları ile değiştirin)
    video1_path = 'ShopAssistant1cor.mpg'  # Kamera 1 videosu (A+B bölgesi)
    video2_path = 'ShopAssistant1front.mpg' 

    
    # Homografi matcher oluştur
    matcher = AutoHomographyMatcher(
        detector_type='SIFT',    # veya 'ORB'
        matcher_type='FLANN',    # veya 'BF'
        min_matches=15           # Düşürüldü, 21 eşleşme için uygun
    )
    
    try:
        # Videoları işle
        results = matcher.process_videos(
            video1_path=video1_path,
            video2_path=video2_path,
            frame1=0,       # Kamera 1'den hangi frame
            frame2=0,       # Kamera 2'den hangi frame
            visualize=True  # Görselleştirme açık
        )
        
        # Sonuçları yazdır
        print("\n=== SONUÇLAR ===")
        print(f"Bulunan anahtar noktalar - Kamera 1: {results['keypoints_img1']}")
        print(f"Bulunan anahtar noktalar - Kamera 2: {results['keypoints_img2']}")
        print(f"İlk eşleştirme sayısı: {results['matches_count']}")
        print(f"RANSAC sonrası güvenilir eşleşme: {results['good_matches_count']}")
        
        print("\nHomografi Matrisi (3x3):")
        print(results['homography_3x3'])
        
        print("\nHomografi Matrisi (4x4):")
        print(results['homography_4x4'])
        
        # Sonuçları kaydet (isteğe bağlı)
        cv2.imwrite('panorama_result.jpg', results['panorama'])
        np.save('homography_matrix.npy', results['homography_3x3'])
        
        print("\nSonuçlar kaydedildi:")
        print("- panorama_result.jpg: Birleştirilmiş panorama")
        print("- homography_matrix.npy: Homografi matrisi")
        
    except Exception as e:
        print(f"HATA: {e}")
        print("\nÖneriler:")
        print("1. Video dosya yollarını kontrol edin")
        print("2. Videoların ortak alanı olduğundan emin olun")
        print("3. Farklı frame numaraları deneyin")
        print("4. Farklı detector tipi deneyin (SIFT/ORB)")
        print("5. min_matches değerini düşürün")

if __name__ == "__main__":
    main()

# Alternatif hızlı test fonksiyonu (görüntü dosyaları ile)
def test_with_images(img1_path, img2_path):
    """İki görüntü dosyası ile hızlı test"""
    
    matcher = AutoHomographyMatcher(detector_type='SIFT', min_matches=30)
    
    # Görüntüleri yükle
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Görüntüler yüklenemedi!")
    
    # Anahtar noktaları tespit et
    kp1, desc1 = matcher.detect_and_compute(img1)
    kp2, desc2 = matcher.detect_and_compute(img2)
    
    # Eşleştirme yap
    matches = matcher.match_features(desc1, desc2)
    
    # Homografi hesapla
    homography, mask, good_matches = matcher.find_homography_ransac(kp1, kp2, matches)
    
    if homography is not None:
        # Panorama oluştur
        panorama, img1_warped, img2_warped = matcher.create_panorama(img1, img2, homography)
        
        # Görselleştir
        matcher.visualize_matches(img1, kp1, img2, kp2, good_matches, "Test Sonucu")
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title('Görüntü 1')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title('Görüntü 2')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title('Panorama')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return homography, panorama
    else:
        print("Homografi hesaplanamadı!")
        return None, None