#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoHomographyMatcher — İki Kamerayı Otomatik Hizalama & Panorama (Tek Dosya)

Bu script, ortak alanı gören iki kamera/videodan (ya da iki görüntüden) manuel işaretleme gerekmeden
SIFT/ORB + FLANN/BF ve RANSAC kullanarak homografi matrisi hesaplar, görüntüleri hizalar ve panorama üretir.

Kurulum:
    pip install opencv-contrib-python numpy matplotlib

Kullanım (Video):
    python autohomography.py video \
        --video1 ShopAssistant1cor.mpg \
        --video2 ShopAssistant1front.mpg \
        --detector SIFT --matcher FLANN \
        --frame1 0 --frame2 0 \
        --min-matches 15 \
        --visualize \
        --outdir outputs

Kullanım (Görüntü):
    python autohomography.py images \
        --img1 kafe_cam1.jpg \
        --img2 kafe_cam2.jpg \
        --detector SIFT --matcher FLANN \
        --min-matches 30 \
        --visualize \
        --outdir outputs

Notlar:
- SIFT kullanmak için `opencv-contrib-python` gerekir.
- FLANN, SIFT için float32; ORB için uint8 (binary) descriptor bekler (kod bunu otomatik ayarlar).
- 4x4 homografi, 3x3 projektif dönüşümün 4x4 matrise gömümüdür (3D anlamı yoktur; pipeline uyumu içindir).
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class AutoHomographyMatcher:
    def __init__(
        self,
        detector_type: str = "SIFT",
        matcher_type: str = "FLANN",
        min_matches: int = 50,
        ratio_thresh: float = 0.8,
        ransac_reproj_threshold: float = 5.0,
        ransac_confidence: float = 0.99,
        ransac_max_iters: int = 5000,
        nfeatures: int = 10000,
    ):
        """
        Otomatik homografi hesaplama ve görüntü hizalama sınıfı.

        Args:
            detector_type: 'SIFT' veya 'ORB'
            matcher_type : 'FLANN' veya 'BF'
            min_matches  : RANSAC öncesi gerekli minimum iyi eşleşme sayısı
            ratio_thresh : Lowe oran testi eşiği (k=2 -> m.distance < ratio * n.distance)
            ransac_reproj_threshold : RANSAC reprojection eşik pikseli
            ransac_confidence       : RANSAC güven seviyesi [0..1]
            ransac_max_iters        : RANSAC maksimum iterasyon
            nfeatures               : SIFT/ORB için özellik sayısı
        """
        self.detector_type = detector_type.upper()
        self.matcher_type = matcher_type.upper()
        self.min_matches = int(min_matches)
        self.ratio_thresh = float(ratio_thresh)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)
        self.ransac_confidence = float(ransac_confidence)
        self.ransac_max_iters = int(ransac_max_iters)

        # Detector seçimi
        if self.detector_type == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError(
                    "SIFT bulunamadı. Lütfen 'opencv-contrib-python' kurulu olduğundan emin olun."
                )
            self.detector = cv2.SIFT_create(nfeatures=int(nfeatures))
        elif self.detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=int(nfeatures))
        else:
            raise ValueError("Detector tipi 'SIFT' veya 'ORB' olmalı")

        # Matcher seçimi
        if self.matcher_type == "FLANN":
            if self.detector_type == "SIFT":
                # SIFT/float32 için FLANN params
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:
                # ORB/binary için FLANN params
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1,
                )
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_type == "BF":
            if self.detector_type == "SIFT":
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError("Matcher tipi 'FLANN' veya 'BF' olmalı")

    # -------------------- Yardımcı Metodlar --------------------

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def extract_frame(self, video_path: Path, frame_number: int = 0) -> np.ndarray:
        """Video dosyasından belirli frame'i çıkart."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise ValueError(
                f"Frame okunamadı: {video_path} (frame: {frame_number})"
            )

        return frame

    def detect_and_compute(self, image: np.ndarray):
        """Görüntüde anahtar noktaları tespit et ve tanımlayıcıları hesapla."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        # Histogram eşitleme ile kontrast artırma
        gray = cv2.equalizeHist(gray)

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        """İki görüntünün tanımlayıcılarını eşleştir + Lowe's ratio test."""
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            d1, d2 = desc1, desc2
            if self.matcher_type == "FLANN":
                # FLANN için descriptor tipini düzelt
                if self.detector_type == "ORB":
                    d1 = d1.astype(np.uint8)
                    d2 = d2.astype(np.uint8)
                else:
                    d1 = d1.astype(np.float32)
                    d2 = d2.astype(np.float32)

            matches_knn = self.matcher.knnMatch(d1, d2, k=2)
            good = []
            for pair in matches_knn:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)
            return good

        except Exception as e:
            print(f"Eşleştirme hatası: {e}")
            return []

    def find_homography_ransac(self, kp1, kp2, matches):
        """RANSAC ile güvenilir homografi matrisi hesapla."""
        if len(matches) < self.min_matches:
            return None, None, []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            dst_pts,
            src_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            confidence=self.ransac_confidence,
            maxIters=self.ransac_max_iters,
        )

        inliers = []
        if mask is not None:
            mask_flat = mask.ravel().tolist()
            inliers = [matches[i] for i, v in enumerate(mask_flat) if v == 1]

        return H, mask, inliers

    def visualize_matches(self, img1, kp1, img2, kp2, matches, title="Eşleşmeler"):
        """Eşleşen noktaları görselleştir."""
        img1c = self._ensure_color(img1)
        img2c = self._ensure_color(img2)

        vis = cv2.drawMatches(
            img1c,
            kp1,
            img2c,
            kp2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (Toplam: {len(matches)} eşleşme)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        return vis

    def create_panorama(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
        """İki görüntüyü homografi ile birleştirip panorama oluştur."""
        img1c = self._ensure_color(img1)
        img2c = self._ensure_color(img2)

        h1, w1 = img1c.shape[:2]
        h2, w2 = img2c.shape[:2]

        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners2, H)

        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners1, transformed_corners), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array(
            [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32
        )

        pano_w = int(x_max - x_min)
        pano_h = int(y_max - y_min)

        img2_warped = cv2.warpPerspective(img2c, translation @ H, (pano_w, pano_h))
        img1_warped = cv2.warpPerspective(img1c, translation, (pano_w, pano_h))

        # Basit birleştirme (maksimum alma). İleri iyileştirme: çok bantlı blending.
        panorama = np.maximum(img1_warped, img2_warped)

        return panorama, img1_warped, img2_warped

    # -------------------- Uçtan Uca Akış --------------------

    def process_videos(
        self,
        video1_path: Path,
        video2_path: Path,
        frame1: int = 0,
        frame2: int = 0,
        visualize: bool = True,
        save_visuals_dir: Path | None = None,
    ):
        """İki video dosyasını işleyerek homografi hesapla."""
        print(f"Video 1'den frame {frame1} çıkartılıyor...")
        img1 = self.extract_frame(video1_path, frame1)

        print(f"Video 2'den frame {frame2} çıkartılıyor...")
        img2 = self.extract_frame(video2_path, frame2)

        print(f"Görüntüler yüklendi: {img1.shape}, {img2.shape}")

        print("Anahtar noktalar tespit ediliyor...")
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)
        print(f"Tespit edilen noktalar - Görüntü 1: {len(kp1)}, Görüntü 2: {len(kp2)}")

        if len(kp1) < 10 or len(kp2) < 10:
            raise ValueError("Yeterli anahtar nokta bulunamadı!")

        print("Özellik eşleştirme yapılıyor...")
        matches = self.match_features(desc1, desc2)
        print(f"İlk eşleştirmede {len(matches)} eşleşme bulundu")

        if visualize and len(matches) > 0:
            vis_all = self.visualize_matches(img1, kp1, img2, kp2, matches, "Tüm Eşleşmeler")
            if save_visuals_dir:
                (save_visuals_dir / "matches_all.jpg").parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_visuals_dir / "matches_all.jpg"), vis_all)

        print("RANSAC ile homografi hesaplanıyor...")
        H, mask, good_matches = self.find_homography_ransac(kp1, kp2, matches)
        if H is None:
            raise ValueError(
                f"Homografi hesaplanamadı! En az {self.min_matches} güvenilir eşleşme gerekli, {len(matches)} bulundu."
            )

        print(f"RANSAC sonrası {len(good_matches)} güvenilir eşleşme kaldı")

        if visualize and len(good_matches) > 0:
            vis_inliers = self.visualize_matches(
                img1, kp1, img2, kp2, good_matches, "Güvenilir Eşleşmeler (RANSAC sonrası)"
            )
            if save_visuals_dir:
                cv2.imwrite(str(save_visuals_dir / "matches_inliers.jpg"), vis_inliers)

        print("Panorama oluşturuluyor...")
        panorama, img1_warped, img2_warped = self.create_panorama(img1, img2, H)

        if visualize:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"Kamera 1 (Frame {frame1})", fontsize=14)
            axes[0, 0].axis("off")

            axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f"Kamera 2 (Frame {frame2})", fontsize=14)
            axes[0, 1].axis("off")

            axes[1, 0].imshow(cv2.cvtColor(img2_warped, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("Kamera 2 - Hizalanmış", fontsize=14)
            axes[1, 0].axis("off")

            axes[1, 1].imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("Birleştirilmiş Panorama", fontsize=14)
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.show()

        # 3x3 -> 4x4 gömme (pipeline uyumu için)
        H4 = np.eye(4, dtype=np.float32)
        H4[:3, :3] = H

        results = {
            "homography_3x3": H,
            "homography_4x4": H4,
            "panorama": panorama,
            "img1_warped": img1_warped,
            "img2_warped": img2_warped,
            "matches_count": len(matches),
            "good_matches_count": len(good_matches),
            "keypoints_img1": len(kp1),
            "keypoints_img2": len(kp2),
        }
        return results

    def process_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        visualize: bool = True,
        save_visuals_dir: Path | None = None,
    ):
        """İki görüntü ile hızlı test."""
        kp1, desc1 = self.detect_and_compute(img1)
        kp2, desc2 = self.detect_and_compute(img2)

        matches = self.match_features(desc1, desc2)

        H, mask, good_matches = self.find_homography_ransac(kp1, kp2, matches)
        if H is None:
            return None, None

        if visualize and len(good_matches) > 0:
            vis_inliers = self.visualize_matches(
                img1, kp1, img2, kp2, good_matches, "Güvenilir Eşleşmeler (RANSAC sonrası)"
            )
            if save_visuals_dir:
                (save_visuals_dir / "matches_inliers.jpg").parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_visuals_dir / "matches_inliers.jpg"), vis_inliers)

        panorama, img1_warped, img2_warped = self.create_panorama(img1, img2, H)
        return H, panorama


# -------------------- CLI --------------------

def save_outputs(results: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Panorama
    pano_path = outdir / "panorama_result.jpg"
    cv2.imwrite(str(pano_path), results["panorama"])

    # Homografi 3x3
    H_path = outdir / "homography_matrix.npy"
    np.save(str(H_path), results["homography_3x3"])

    # Warped görseller
    cv2.imwrite(str(outdir / "img1_warped.jpg"), results["img1_warped"])
    cv2.imwrite(str(outdir / "img2_warped.jpg"), results["img2_warped"])

    print("\n=== SONUÇLAR ===")
    print(f"Bulunan anahtar noktalar - Kamera 1: {results['keypoints_img1']}")
    print(f"Bulunan anahtar noktalar - Kamera 2: {results['keypoints_img2']}")
    print(f"İlk eşleştirme sayısı: {results['matches_count']}")
    print(f"RANSAC sonrası güvenilir eşleşme: {results['good_matches_count']}")
    print("\nHomografi Matrisi (3x3):")
    print(results["homography_3x3"])
    print("\nHomografi Matrisi (4x4):")
    print(results["homography_4x4"])
    print("\nKaydedilen çıktılar:")
    print(f"- {pano_path.name} (panorama)")
    print(f"- {H_path.name} (homografi 3x3)")
    print("- img1_warped.jpg")
    print("- img2_warped.jpg")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoHomographyMatcher — iki kaynaktan homografi ve panorama üretimi"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # Ortak argümanlar
    def add_common_args(sp):
        sp.add_argument("--detector", choices=["SIFT", "ORB"], default="SIFT")
        sp.add_argument("--matcher", choices=["FLANN", "BF"], default="FLANN")
        sp.add_argument("--min-matches", type=int, default=50)
        sp.add_argument("--ratio", type=float, default=0.8, help="Lowe oran testi eşiği")
        sp.add_argument("--ransac-th", type=float, default=5.0, help="RANSAC reproj eşiği (px)")
        sp.add_argument("--ransac-conf", type=float, default=0.99, help="RANSAC güven seviyesi")
        sp.add_argument("--ransac-iters", type=int, default=5000, help="RANSAC max iterasyon")
        sp.add_argument("--nfeatures", type=int, default=10000, help="Özellik sayısı (SIFT/ORB)")
        sp.add_argument("--visualize", action="store_true", help="Görselleştirme")
        sp.add_argument("--outdir", type=Path, default=Path("outputs"), help="Çıktı klasörü")

    # Video modu
    sp_video = sub.add_parser("video", help="İki videodan homografi & panorama üret")
    add_common_args(sp_video)
    sp_video.add_argument("--video1", type=Path, required=True)
    sp_video.add_argument("--video2", type=Path, required=True)
    sp_video.add_argument("--frame1", type=int, default=0)
    sp_video.add_argument("--frame2", type=int, default=0)

    # Görüntü modu
    sp_img = sub.add_parser("images", help="İki görüntüden homografi & panorama üret")
    add_common_args(sp_img)
    sp_img.add_argument("--img1", type=Path, required=True)
    sp_img.add_argument("--img2", type=Path, required=True)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    matcher = AutoHomographyMatcher(
        detector_type=args.detector,
        matcher_type=args.matcher,
        min_matches=args.min_matches,
        ratio_thresh=args.ratio,
        ransac_reproj_threshold=args.ransac_th,
        ransac_confidence=args.ransac_conf,
        ransac_max_iters=args.ransac_iters,
        nfeatures=args.nfeatures,
    )

    outdir: Path = args.outdir

    try:
        if args.mode == "video":
            results = matcher.process_videos(
                video1_path=args.video1,
                video2_path=args.video2,
                frame1=args.frame1,
                frame2=args.frame2,
                visualize=args.visualize,
                save_visuals_dir=outdir if args.visualize else None,
            )
            save_outputs(results, outdir)

        elif args.mode == "images":
            img1 = cv2.imread(str(args.img1))
            img2 = cv2.imread(str(args.img2))
            if img1 is None or img2 is None:
                raise ValueError("Görüntü(ler) yüklenemedi! Dosya yollarını kontrol edin.")

            H, pano = matcher.process_images(
                img1=img1,
                img2=img2,
                visualize=args.visualize,
                save_visuals_dir=outdir if args.visualize else None,
            )

            if H is None:
                raise RuntimeError("Homografi hesaplanamadı (yeterli güvenilir eşleşme yok).")

            # 4x4 gömme
            H4 = np.eye(4, dtype=np.float32)
            H4[:3, :3] = H

            results = {
                "homography_3x3": H,
                "homography_4x4": H4,
                "panorama": pano,
                "img1_warped": np.zeros_like(pano),  # video akışındaki gibi placeholder
                "img2_warped": np.zeros_like(pano),
                "matches_count": -1,
                "good_matches_count": -1,
                "keypoints_img1": -1,
                "keypoints_img2": -1,
            }
            save_outputs(results, outdir)

    except Exception as e:
        print(f"\nHATA: {e}")
        print("\nÖneriler:")
        print("1) Dosya yollarını/izinlerini kontrol edin")
        print("2) Videoların/görüntülerin ortak alanı olduğundan emin olun")
        print("3) Farklı frame numaraları deneyin (video modu)")
        print("4) Farklı dedektör/eşleştirici deneyin (SIFT/ORB, FLANN/BF)")
        print("5) 'min-matches' değerini düşürün ve 'ratio' değerini 0.85'e çıkarın")
        print("6) 'nfeatures' değerini artırın; çözünürlüğü/kontrastı iyileştirin")


if __name__ == "__main__":
    main()
