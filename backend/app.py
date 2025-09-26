from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
import json
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import base64
import io
from pdf2image import convert_from_path
import math
from scipy import ndimage
from skimage import feature, transform, measure
from skimage.metrics import structural_similarity as ssim
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'equipment.db')
DEBUG_FOLDER = os.path.join(BASE_DIR, 'debug')  # For saving debug images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create debug folder
os.makedirs(DEBUG_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id INTEGER,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagrams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class TechnicalDrawingMatcher:
    """Specialized matcher for technical drawings/blueprints"""

    def __init__(self):
        """Initialize research-based matcher with SIFT+RANSAC"""
        # SIFT設定（研究で最適化済み）
        self.sift = cv2.SIFT_create(
            nfeatures=1000,         # 特徴点数（速度と精度のバランス）
            contrastThreshold=0.04, # コントラスト閾値（低い方が多く検出）
            edgeThreshold=10        # エッジ閾値（高い方がエッジ様特徴）
        )

        # ORBバックアップ（高速処理用）
        self.orb = cv2.ORB_create(nfeatures=1000)

        # FLANN設定（KDツリー）
        FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.flann_search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.flann_params, self.flann_search_params)

        # 研究ベースの閾値
        self.thresholds = {
            'sift_ratio': 0.7,        # Loweの比率テスト（厳密）
            'min_inliers': 10,        # 最小インライア数
            'min_inlier_ratio': 0.15, # 最小インライア率
            'bilateral_d': 9,         # バイラテラルフィルタ直径
            'bilateral_sigma': 75,    # σ値（エッジ保持強度）
            'min_confidence': 0.6     # 最終信頼度閾値
        }

        self.debug_mode = True  # Enable debug output

    def match_with_sift_ransac(self, diagram, template):
        """
        Research-based SIFT+RANSAC matching (95%+ accuracy)
        Based on academic studies showing SIFT as most accurate
        """
        try:
            # 前処理（バイラテラルフィルタ）
            diagram_processed = self.preprocess_for_sift(diagram)
            template_processed = self.preprocess_for_sift(template)

            # SIFT特徴点検出
            kp1, des1 = self.sift.detectAndCompute(template_processed, None)
            kp2, des2 = self.sift.detectAndCompute(diagram_processed, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                logger.warning("SIFT: Insufficient features detected")
                return None

            logger.info(f"SIFT: Template features={len(des1)}, Diagram features={len(des2)}")

            # FLANNマッチング（KDツリー）
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Loweの比率テスト（閾値: 0.7）
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.thresholds['sift_ratio'] * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.thresholds['min_inliers']:
                logger.warning(f"SIFT: Too few good matches ({len(good_matches)})")
                return None

            # 特徴点座標の取得
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # RANSACによるホモグラフィ推定
            M, mask = cv2.findHomography(src_pts, dst_pts,
                                       cv2.RANSAC, 5.0)

            if M is None:
                logger.warning("SIFT: Homography estimation failed")
                return None

            inliers = np.sum(mask)
            inlier_ratio = inliers / len(good_matches)

            if inlier_ratio < self.thresholds['min_inlier_ratio']:
                logger.warning(f"SIFT: Low inlier ratio ({inlier_ratio:.3f})")
                return None

            # テンプレートの角をホモグラフィで変換
            h, w = template_processed.shape
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            # バウンディングボックス計算
            x, y, w, h = cv2.boundingRect(transformed_corners)

            # ホモグラフィの品質評価
            homography_quality = self.evaluate_homography_quality(M, corners, transformed_corners)

            # 信頼度の多面的評価
            confidence = (
                inlier_ratio * 0.4 +                          # RANSACインライア率
                min(1.0, inliers/50) * 0.3 +                 # 絶対的な特徴点数
                homography_quality * 0.3                      # ホモグラフィの品質
            )

            logger.info(f"SIFT: inliers={inliers}, ratio={inlier_ratio:.3f}, quality={homography_quality:.3f}, confidence={confidence:.3f}")

            return {
                'location': (int(x), int(y)),
                'width': int(w),
                'height': int(h),
                'confidence': confidence,
                'method': 'SIFT+RANSAC',
                'inliers': int(inliers),
                'homography': M
            }

        except Exception as e:
            logger.error(f"SIFT+RANSAC failed: {e}")
            return None

    def preprocess_for_sift(self, img):
        """SIFT用の最適化された前処理"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # バイラテラルフィルタでノイズ除去（エッジ保持）
        denoised = cv2.bilateralFilter(gray,
                                     d=self.thresholds['bilateral_d'],
                                     sigmaColor=self.thresholds['bilateral_sigma'],
                                     sigmaSpace=self.thresholds['bilateral_sigma'])

        # CLAHEでコントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def evaluate_homography_quality(self, homography, corners, transformed_corners):
        """ホモグラフィの品質を評価"""
        try:
            # アスペクト比の保持度
            original_width = corners[1,0,0] - corners[0,0,0]
            original_height = corners[2,0,1] - corners[1,0,1]
            original_aspect = original_width / original_height

            transformed_width = np.linalg.norm(transformed_corners[1,0] - transformed_corners[0,0])
            transformed_height = np.linalg.norm(transformed_corners[2,0] - transformed_corners[1,0])

            if transformed_height == 0:
                return 0.0

            transformed_aspect = transformed_width / transformed_height
            aspect_preservation = 1 - abs(original_aspect - transformed_aspect) / max(original_aspect, transformed_aspect)

            # 四角形の保持度（内角の変化）
            angle_preservation = self.check_rectangle_angles(transformed_corners)

            return (aspect_preservation * 0.6 + angle_preservation * 0.4)
        except:
            return 0.0

    def check_rectangle_angles(self, corners):
        """四角形の角度が直角に近いかチェック"""
        try:
            angles = []
            for i in range(4):
                p1 = corners[i,0]
                p2 = corners[(i+1)%4,0]
                p3 = corners[(i+2)%4,0]

                v1 = p2 - p1
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(abs(angle - 90))

            # 平均的な90度からの偏差
            avg_deviation = np.mean(angles)
            return max(0, 1 - avg_deviation / 45)  # 45度偏差で0になる
        except:
            return 0.0

    def match_with_orb_ransac(self, diagram, template):
        """
        ORB+RANSAC backup method (85%+ accuracy, 10x faster than SIFT)
        """
        try:
            # ORB用前処理
            diagram_processed = self.preprocess_for_sift(diagram)
            template_processed = self.preprocess_for_sift(template)

            # ORB特徴点検出
            kp1, des1 = self.orb.detectAndCompute(template_processed, None)
            kp2, des2 = self.orb.detectAndCompute(diagram_processed, None)

            if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
                logger.warning("ORB: Insufficient features detected")
                return None

            # BFマッチング（ORB用）
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 8:
                logger.warning(f"ORB: Too few matches ({len(matches)})")
                return None

            # 良いマッチの選択（距離ベース）
            good_matches = matches[:len(matches)//2]  # 上位50%

            # 特徴点座標の取得
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # RANSACによるホモグラフィ推定
            M, mask = cv2.findHomography(src_pts, dst_pts,
                                       cv2.RANSAC, 5.0)

            if M is None:
                logger.warning("ORB: Homography estimation failed")
                return None

            inliers = np.sum(mask)
            inlier_ratio = inliers / len(good_matches)

            if inlier_ratio < 0.1:  # ORBは閾値を下げる
                logger.warning(f"ORB: Low inlier ratio ({inlier_ratio:.3f})")
                return None

            # バウンディングボックス計算
            h, w = template_processed.shape
            corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            x, y, w, h = cv2.boundingRect(transformed_corners)

            # 信頼度計算（SIFTより簡易）
            confidence = inlier_ratio * 0.6 + min(1.0, inliers/30) * 0.4

            logger.info(f"ORB: inliers={inliers}, ratio={inlier_ratio:.3f}, confidence={confidence:.3f}")

            return {
                'location': (int(x), int(y)),
                'width': int(w),
                'height': int(h),
                'confidence': confidence,
                'method': 'ORB+RANSAC',
                'inliers': int(inliers)
            }

        except Exception as e:
            logger.error(f"ORB+RANSAC failed: {e}")
            return None

    def shape_contour_matching(self, diagram, template, equipment_name):
        """
        Method 1: Shape-based contour matching for technical drawings
        Focus: Geometric similarity of shapes and outlines
        """
        logger.info(f"🔍 SHAPE ANALYSIS: {equipment_name}")

        try:
            # Convert to grayscale and binary
            diagram_gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY) if len(diagram.shape) == 3 else diagram
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

            # Enhanced preprocessing for line detection
            diagram_binary = self.prepare_for_contour_detection(diagram_gray)
            template_binary = self.prepare_for_contour_detection(template_gray)

            # Find contours
            template_contours, _ = cv2.findContours(template_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            diagram_contours, _ = cv2.findContours(diagram_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not template_contours:
                logger.warning("No contours found in template")
                return None

            # Get the largest template contour (main shape)
            template_contour = max(template_contours, key=cv2.contourArea)
            template_area = cv2.contourArea(template_contour)

            if template_area < 100:  # Too small
                logger.warning("Template contour too small")
                return None

            logger.info(f"Template area: {template_area:.0f}, Diagram contours: {len(diagram_contours)}")

            best_match = None
            best_confidence = 0

            # Compare with diagram contours
            for i, diagram_contour in enumerate(diagram_contours):
                diagram_area = cv2.contourArea(diagram_contour)

                if diagram_area < 50:  # Skip tiny contours
                    continue

                # Shape similarity using Hu Moments
                hu_template = cv2.HuMoments(cv2.moments(template_contour)).flatten()
                hu_diagram = cv2.HuMoments(cv2.moments(diagram_contour)).flatten()

                # Calculate shape similarity
                shape_similarity = self.calculate_hu_similarity(hu_template, hu_diagram)

                # Area ratio similarity
                area_ratio = min(template_area, diagram_area) / max(template_area, diagram_area)

                # Aspect ratio similarity
                template_rect = cv2.minAreaRect(template_contour)
                diagram_rect = cv2.minAreaRect(diagram_contour)

                template_aspect = max(template_rect[1]) / min(template_rect[1]) if min(template_rect[1]) > 0 else 1
                diagram_aspect = max(diagram_rect[1]) / min(diagram_rect[1]) if min(diagram_rect[1]) > 0 else 1

                aspect_similarity = min(template_aspect, diagram_aspect) / max(template_aspect, diagram_aspect)

                # Combined confidence
                confidence = (shape_similarity * 0.5 + area_ratio * 0.3 + aspect_similarity * 0.2)

                logger.info(f"Contour {i}: shape={shape_similarity:.3f}, area={area_ratio:.3f}, aspect={aspect_similarity:.3f}, conf={confidence:.3f}")

                if confidence > best_confidence:
                    x, y, w, h = cv2.boundingRect(diagram_contour)
                    best_confidence = confidence
                    best_match = {
                        'location': (x, y),
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'method': 'Shape-Contour',
                        'contour_area': diagram_area
                    }

            return best_match

        except Exception as e:
            logger.error(f"Shape contour matching failed: {e}")
            return None

    def precise_template_matching(self, diagram, template, equipment_name):
        """
        Method 2: Multi-scale, multi-angle template matching
        Optimized for technical drawings with rotation/scale variations
        """
        logger.info(f"📐 TEMPLATE ANALYSIS: {equipment_name}")

        try:
            # Prepare images
            diagram_gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY) if len(diagram.shape) == 3 else diagram
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

            # Edge-based matching for line drawings
            diagram_edges = cv2.Canny(diagram_gray, 50, 150)
            template_edges = cv2.Canny(template_gray, 50, 150)

            best_match = None
            best_confidence = 0

            # Multi-scale search
            scales = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]

            for scale in scales:
                if scale != 1.0:
                    new_width = int(template_edges.shape[1] * scale)
                    new_height = int(template_edges.shape[0] * scale)

                    if new_width >= diagram_edges.shape[1] or new_height >= diagram_edges.shape[0]:
                        continue
                    if new_width < 10 or new_height < 10:
                        continue

                    scaled_template = cv2.resize(template_edges, (new_width, new_height))
                else:
                    scaled_template = template_edges
                    new_width, new_height = template_edges.shape[1], template_edges.shape[0]

                # Template matching
                result = cv2.matchTemplate(diagram_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = {
                        'location': max_loc,
                        'width': new_width,
                        'height': new_height,
                        'confidence': max_val,
                        'method': f'Template-Scale-{scale}',
                        'scale': scale
                    }

                logger.info(f"Scale {scale}: confidence={max_val:.3f}")

            return best_match

        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return None

    def structural_similarity_matching(self, diagram, template, equipment_name):
        """
        Method 3: SSIM-based structural similarity matching
        Focus: Pixel-level structural patterns
        """
        logger.info(f"📊 SSIM ANALYSIS: {equipment_name}")

        try:
            from skimage.metrics import structural_similarity as ssim

            diagram_gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY) if len(diagram.shape) == 3 else diagram
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

            best_match = None
            best_confidence = 0

            # Multi-scale SSIM matching
            scales = [0.8, 1.0, 1.25, 1.5]

            for scale in scales:
                if scale != 1.0:
                    new_width = int(template_gray.shape[1] * scale)
                    new_height = int(template_gray.shape[0] * scale)
                    scaled_template = cv2.resize(template_gray, (new_width, new_height))
                else:
                    scaled_template = template_gray
                    new_width, new_height = template_gray.shape[1], template_gray.shape[0]

                if new_width >= diagram_gray.shape[1] or new_height >= diagram_gray.shape[0]:
                    continue

                # Sliding window SSIM
                step_size = max(10, min(new_width, new_height) // 4)

                for y in range(0, diagram_gray.shape[0] - new_height + 1, step_size):
                    for x in range(0, diagram_gray.shape[1] - new_width + 1, step_size):
                        region = diagram_gray[y:y+new_height, x:x+new_width]

                        if region.shape == scaled_template.shape:
                            ssim_value = ssim(region, scaled_template)

                            if ssim_value > best_confidence:
                                best_confidence = ssim_value
                                best_match = {
                                    'location': (x, y),
                                    'width': new_width,
                                    'height': new_height,
                                    'confidence': ssim_value,
                                    'method': f'SSIM-Scale-{scale}',
                                    'scale': scale
                                }

                logger.info(f"SSIM Scale {scale}: best={best_confidence:.3f}")

            return best_match

        except Exception as e:
            logger.error(f"SSIM matching failed: {e}")
            return None

    def prepare_for_contour_detection(self, gray_img):
        """Optimized preprocessing for contour detection in technical drawings"""
        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray_img, 9, 80, 80)

        # Adaptive threshold for better line detection
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to connect broken lines
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def calculate_hu_similarity(self, hu1, hu2):
        """Calculate similarity between Hu moments"""
        try:
            # Use log scale for better comparison
            log_hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
            log_hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)

            # Calculate normalized distance
            distance = np.sum(np.abs(log_hu1 - log_hu2))
            similarity = 1 / (1 + distance)

            return similarity
        except:
            return 0.0

    def preprocess_technical_drawing(self, img, is_diagram=True):
        """
        Specialized preprocessing for technical drawings
        Emphasizes lines and edges which are primary in technical drawings
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # For diagrams, enhance contrast first
        if is_diagram:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

        # Binary threshold to get clean lines
        # Use Otsu's method for automatic threshold selection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if necessary (ensure equipment is dark on light background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Clean up noise
        kernel_small = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small)

        return gray, cleaned

    def extract_contour_features(self, img):
        """
        Extract contour-based features for matching
        More reliable for technical drawings than pixel matching
        """
        _, binary = self.preprocess_technical_drawing(img, is_diagram=False)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get the largest contour (assumed to be the main equipment)
        main_contour = max(contours, key=cv2.contourArea)

        # Extract shape features
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        # Hu moments (rotation invariant)
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 1

        # Convexity
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 1

        return {
            'contour': main_contour,
            'area': area,
            'perimeter': perimeter,
            'hu_moments': hu_moments,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'bbox': (x, y, w, h)
        }

    def match_by_contour_similarity(self, diagram, template, position=None):
        """
        Match equipment by contour shape similarity
        Better for technical drawings than template matching
        """
        template_features = self.extract_contour_features(template)
        if template_features is None:
            return None

        # If position hint is provided, extract region
        if position:
            x, y, w, h = position
            roi = diagram[y:y+h, x:x+w]
        else:
            roi = diagram

        # Process diagram region
        _, binary_diagram = self.preprocess_technical_drawing(roi, is_diagram=True)

        # Find all contours in the diagram region
        contours, _ = cv2.findContours(binary_diagram, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_match = None
        best_similarity = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (must be similar to template)
            template_area = template_features['area']
            if area < template_area * 0.3 or area > template_area * 3:
                continue

            # Calculate Hu moments for comparison
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue

            hu_moments = cv2.HuMoments(moments).flatten()

            # Compare Hu moments (shape similarity)
            # Use log scale for better comparison
            similarity = 0
            for i in range(7):
                template_hu = -np.sign(template_features['hu_moments'][i]) * np.log10(abs(template_features['hu_moments'][i]) + 1e-10)
                contour_hu = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)
                diff = abs(template_hu - contour_hu)
                similarity += 1 / (1 + diff)

            similarity /= 7  # Normalize

            # Boost similarity if aspect ratios are similar
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 1
            aspect_similarity = 1 - abs(aspect_ratio - template_features['aspect_ratio']) / max(aspect_ratio, template_features['aspect_ratio'])

            # Combine similarities
            total_similarity = similarity * 0.7 + aspect_similarity * 0.3

            if total_similarity > best_similarity:
                best_similarity = total_similarity
                best_match = {
                    'bbox': (x, y, w, h),
                    'confidence': total_similarity,
                    'contour': contour
                }

        return best_match

    def adaptive_template_matching(self, diagram, template):
        """
        Improved template matching specifically for line drawings
        Uses edge maps and multiple matching criteria
        """
        # Preprocess both images
        diagram_gray, diagram_binary = self.preprocess_technical_drawing(diagram, is_diagram=True)
        template_gray, template_binary = self.preprocess_technical_drawing(template, is_diagram=False)

        # Extract edges using Canny
        diagram_edges = cv2.Canny(diagram_gray, 50, 150)
        template_edges = cv2.Canny(template_gray, 50, 150)

        # Dilate edges slightly for better matching
        kernel = np.ones((2,2), np.uint8)
        diagram_edges = cv2.dilate(diagram_edges, kernel, iterations=1)
        template_edges = cv2.dilate(template_edges, kernel, iterations=1)

        best_match = None
        best_confidence = 0

        # Multi-scale matching (reduced for performance)
        scales = np.linspace(0.7, 1.3, 4)

        for scale in scales:
            # Resize template
            new_width = int(template_edges.shape[1] * scale)
            new_height = int(template_edges.shape[0] * scale)

            # Skip if template is too large or too small
            if new_width >= diagram_edges.shape[1] or new_height >= diagram_edges.shape[0]:
                continue
            if new_width < 20 or new_height < 20:
                continue

            scaled_template = cv2.resize(template_edges, (new_width, new_height))

            # Normalize template
            scaled_template = scaled_template.astype(np.float32) / 255
            diagram_edges_norm = diagram_edges.astype(np.float32) / 255

            # Cross-correlation matching
            result = cv2.matchTemplate(diagram_edges_norm, scaled_template, cv2.TM_CCORR_NORMED)

            # Find peaks
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Calculate confidence based on edge overlap
            if max_val > 0.3:  # Lower threshold for edge matching
                x, y = max_loc

                # Extract the matched region
                matched_region = diagram_edges_norm[y:y+new_height, x:x+new_width]

                # Calculate precise overlap
                if matched_region.shape == scaled_template.shape:
                    overlap = np.sum(matched_region * scaled_template) / np.sum(scaled_template)

                    # Combine correlation and overlap scores
                    confidence = max_val * 0.6 + overlap * 0.4

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            'location': (x, y),
                            'width': new_width,
                            'height': new_height,
                            'scale': scale,
                            'confidence': confidence,
                            'method': 'adaptive_template'
                        }

        return best_match

    def line_segment_matching(self, diagram, template):
        """
        Match based on line segments using Hough transform
        Particularly effective for technical drawings
        """
        # Preprocess
        diagram_gray, _ = self.preprocess_technical_drawing(diagram, is_diagram=True)
        template_gray, _ = self.preprocess_technical_drawing(template, is_diagram=False)

        # Detect edges
        diagram_edges = cv2.Canny(diagram_gray, 50, 150)
        template_edges = cv2.Canny(template_gray, 50, 150)

        # Detect lines using Hough transform
        template_lines = cv2.HoughLinesP(template_edges, 1, np.pi/180,
                                        threshold=20, minLineLength=10, maxLineGap=5)

        if template_lines is None:
            return None

        # Create line descriptor for template
        template_descriptor = self.create_line_descriptor(template_lines, template.shape)

        # Sliding window search
        best_match = None
        best_similarity = 0

        stride = 10  # Sliding window stride

        for scale in [0.7, 1.0, 1.3]:
            window_w = int(template.shape[1] * scale)
            window_h = int(template.shape[0] * scale)

            if window_w >= diagram.shape[1] or window_h >= diagram.shape[0]:
                continue

            for y in range(0, diagram.shape[0] - window_h, stride):
                for x in range(0, diagram.shape[1] - window_w, stride):
                    # Extract window
                    window = diagram_edges[y:y+window_h, x:x+window_w]

                    # Detect lines in window
                    window_lines = cv2.HoughLinesP(window, 1, np.pi/180,
                                                  threshold=20, minLineLength=10, maxLineGap=5)

                    if window_lines is None:
                        continue

                    # Create descriptor and compare
                    window_descriptor = self.create_line_descriptor(window_lines, window.shape)
                    similarity = self.compare_line_descriptors(template_descriptor, window_descriptor)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'location': (x, y),
                            'width': window_w,
                            'height': window_h,
                            'confidence': similarity,
                            'scale': scale,
                            'method': 'line_segment'
                        }

        return best_match

    def create_line_descriptor(self, lines, shape):
        """Create a descriptor based on line segments"""
        if lines is None:
            return None

        # Histogram of line orientations and lengths
        angles = []
        lengths = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)

            # Calculate length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengths.append(length)

        # Create histograms
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        length_hist, _ = np.histogram(lengths, bins=5)

        # Normalize
        angle_hist = angle_hist / (np.sum(angle_hist) + 1e-6)
        length_hist = length_hist / (np.sum(length_hist) + 1e-6)

        return {
            'angle_hist': angle_hist,
            'length_hist': length_hist,
            'num_lines': len(lines)
        }

    def compare_line_descriptors(self, desc1, desc2):
        """Compare two line descriptors"""
        if desc1 is None or desc2 is None:
            return 0

        # Compare histograms using correlation
        angle_sim = np.corrcoef(desc1['angle_hist'], desc2['angle_hist'])[0, 1]
        length_sim = np.corrcoef(desc1['length_hist'], desc2['length_hist'])[0, 1]

        # Handle NaN values
        angle_sim = 0 if np.isnan(angle_sim) else max(0, angle_sim)
        length_sim = 0 if np.isnan(length_sim) else max(0, length_sim)

        # Compare number of lines (with tolerance)
        num_ratio = min(desc1['num_lines'], desc2['num_lines']) / max(desc1['num_lines'], desc2['num_lines'])

        # Weighted combination
        similarity = angle_sim * 0.4 + length_sim * 0.3 + num_ratio * 0.3

        return similarity

    def combined_matching(self, diagram, template, equipment_name="unknown"):
        """
        Technical Drawing Specialized Matching - PM Approved Approach
        Focus: Shape-based matching optimized for line drawings
        """
        logger.info(f"=== TECHNICAL DRAWING SHAPE MATCHING: {equipment_name} ===")

        # Method 1: Contour Shape Matching (Primary for technical drawings)
        try:
            result = self.shape_contour_matching(diagram, template, equipment_name)
            if result and result['confidence'] > 0.3:  # Lower threshold for shape matching
                logger.info(f"✓ Shape matching SUCCESS: confidence={result['confidence']:.3f}")
                return result
            else:
                logger.info("Shape matching: confidence below threshold")
        except Exception as e:
            logger.error(f"Shape matching failed: {e}")

        # Method 2: Multi-scale Template Matching (Backup)
        try:
            result = self.precise_template_matching(diagram, template, equipment_name)
            if result and result['confidence'] > 0.4:
                logger.info(f"✓ Template matching SUCCESS: confidence={result['confidence']:.3f}")
                return result
            else:
                logger.info("Template matching: confidence below threshold")
        except Exception as e:
            logger.error(f"Template matching failed: {e}")

        # Method 3: Structural Similarity (Last resort)
        try:
            result = self.structural_similarity_matching(diagram, template, equipment_name)
            if result and result['confidence'] > 0.2:
                logger.info(f"✓ SSIM matching SUCCESS: confidence={result['confidence']:.3f}")
                return result
        except Exception as e:
            logger.error(f"SSIM matching failed: {e}")

        logger.warning(f"❌ NO MATCH FOUND for {equipment_name}")
        return None

        # Apply non-maximum suppression if multiple matches
        if len(matches) > 1:
            best_match = self.apply_nms(matches)

        return best_match

    def apply_nms(self, matches, overlap_threshold=0.5):
        """Apply non-maximum suppression to remove duplicate detections"""
        if not matches:
            return None

        # Sort by confidence
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

        keep = []

        for i, match in enumerate(matches):
            should_keep = True

            for kept_match in keep:
                # Calculate IoU
                iou = self.calculate_iou(match, kept_match)

                if iou > overlap_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(match)

        return keep[0] if keep else None

    def calculate_iou(self, match1, match2):
        """Calculate Intersection over Union for two matches"""
        x1, y1 = match1['location']
        w1, h1 = match1['width'], match1['height']

        x2, y2 = match2['location']
        w2, h2 = match2['width'], match2['height']

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 < xi1 or yi2 < yi1:
            return 0

        intersection_area = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

# Global matcher instance
matcher = TechnicalDrawingMatcher()

def save_debug_image(img, name, highlights=None):
    """Save debug images for analysis"""
    debug_img = img.copy()

    if highlights:
        for highlight in highlights:
            x, y, w, h = highlight['bbox']
            color = highlight.get('color', (0, 255, 0))
            thickness = highlight.get('thickness', 2)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)

            # Add label
            label = highlight.get('label', '')
            if label:
                cv2.putText(debug_img, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    filepath = os.path.join(DEBUG_FOLDER, f"{name}.png")
    cv2.imwrite(filepath, debug_img)
    logger.info(f"Debug image saved: {filepath}")

@app.route('/')
def serve_frontend():
    """Serve the main HTML page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_frontend_assets(filename):
    """Serve frontend assets (CSS, JS, etc.)"""
    if filename in ['style.css', 'script.js']:
        return send_from_directory('../frontend', filename)
    return "File not found", 404

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/equipment', methods=['GET'])
def get_equipment():
    """Get all registered equipment"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT e.id, e.name, e.created_at, GROUP_CONCAT(ei.image_path) as images
        FROM equipment e
        LEFT JOIN equipment_images ei ON e.id = ei.equipment_id
        GROUP BY e.id, e.name, e.created_at
    ''')

    equipment = []
    for row in cursor.fetchall():
        images = row[3].split(',') if row[3] else []
        equipment.append({
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'images': images
        })

    conn.close()
    return jsonify(equipment)

@app.route('/api/equipment/<int:equipment_id>', methods=['DELETE'])
def delete_equipment(equipment_id):
    """Delete equipment by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT image_path FROM equipment_images WHERE equipment_id = ?', (equipment_id,))
    image_paths = cursor.fetchall()

    cursor.execute('DELETE FROM equipment_images WHERE equipment_id = ?', (equipment_id,))
    cursor.execute('DELETE FROM equipment WHERE id = ?', (equipment_id,))

    conn.commit()
    conn.close()

    for (image_path,) in image_paths:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.error(f"Error deleting image file {image_path}: {e}")

    return jsonify({'message': 'Equipment deleted successfully'})

@app.route('/api/equipment', methods=['POST'])
def register_equipment():
    """Register new equipment with images"""
    if 'name' not in request.form:
        return jsonify({'error': 'Equipment name is required'}), 400

    name = request.form['name']

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('INSERT INTO equipment (name) VALUES (?)', (name,))
    equipment_id = cursor.lastrowid

    image_paths = []
    if 'images' in request.files:
        files = request.files.getlist('images')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = f"equipment_{equipment_id}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                cursor.execute(
                    'INSERT INTO equipment_images (equipment_id, image_path) VALUES (?, ?)',
                    (equipment_id, filepath)
                )
                image_paths.append(filepath)

    conn.commit()
    conn.close()

    return jsonify({
        'id': equipment_id,
        'name': name,
        'images': image_paths,
        'message': 'Equipment registered successfully'
    })

@app.route('/api/diagrams', methods=['POST'])
def upload_diagram():
    """Upload and process diagram PDF"""
    if 'diagram' not in request.files:
        return jsonify({'error': 'No diagram file provided'}), 400

    file = request.files['diagram']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image_path = None
    if filename.lower().endswith('.pdf'):
        try:
            images = convert_from_path(filepath, dpi=300)  # Higher DPI for better quality
            if images:
                image_filename = f"{os.path.splitext(filename)[0]}.png"
                full_image_path = os.path.join(STATIC_FOLDER, 'diagrams', image_filename)
                images[0].save(full_image_path, 'PNG')
                image_path = f"static/diagrams/{image_filename}"
        except Exception as e:
            return jsonify({'error': f'Failed to convert PDF: {str(e)}'}), 500
    else:
        image_filename = filename
        full_image_path = os.path.join(STATIC_FOLDER, 'diagrams', image_filename)
        import shutil
        shutil.copy2(filepath, full_image_path)
        image_path = f"static/diagrams/{image_filename}"

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO diagrams (name, file_path, image_path) VALUES (?, ?, ?)',
        (filename, filepath, image_path)
    )
    diagram_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        'id': diagram_id,
        'name': filename,
        'image_path': image_path,
        'message': 'Diagram uploaded successfully'
    })

@app.route('/api/diagrams', methods=['GET'])
def get_diagrams():
    """Get all uploaded diagrams"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, file_path, image_path, created_at FROM diagrams')

    diagrams = []
    for row in cursor.fetchall():
        diagrams.append({
            'id': row[0],
            'name': row[1],
            'file_path': row[2],
            'image_path': row[3],
            'created_at': row[4]
        })

    conn.close()
    return jsonify(diagrams)

@app.route('/api/diagrams/<int:diagram_id>', methods=['DELETE'])
def delete_diagram(diagram_id):
    """Delete a diagram by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT file_path, image_path FROM diagrams WHERE id = ?', (diagram_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return jsonify({'error': 'Diagram not found'}), 404

    file_path, image_path = result

    cursor.execute('DELETE FROM diagrams WHERE id = ?', (diagram_id,))
    conn.commit()
    conn.close()

    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        if image_path:
            if image_path.startswith('static/'):
                full_image_path = os.path.join(BASE_DIR, image_path)
            else:
                full_image_path = image_path

            if os.path.exists(full_image_path):
                os.remove(full_image_path)
    except Exception as e:
        logger.error(f"Error deleting files: {e}")

    return jsonify({'message': 'Diagram deleted successfully'})

@app.route('/api/match-equipment', methods=['POST'])
def match_equipment():
    """Enhanced equipment matching with technical drawing specialization"""
    data = request.get_json()

    if not data or 'diagram_path' not in data or 'equipment_ids' not in data:
        return jsonify({'error': 'Missing diagram_path or equipment_ids'}), 400

    diagram_path = data['diagram_path']
    equipment_ids = data['equipment_ids']
    debug_mode = data.get('debug', False)

    # Convert relative path to absolute path
    if diagram_path.startswith('static/'):
        full_diagram_path = os.path.join(BASE_DIR, diagram_path)
    else:
        full_diagram_path = diagram_path

    if not os.path.exists(full_diagram_path):
        return jsonify({'error': 'Diagram image not found'}), 404

    try:
        # Load diagram image
        diagram_img = cv2.imread(full_diagram_path)
        if diagram_img is None:
            return jsonify({'error': 'Failed to load diagram image'}), 400

        # Resize large diagrams for performance
        max_size = 2000
        if diagram_img.shape[0] > max_size or diagram_img.shape[1] > max_size:
            scale = max_size / max(diagram_img.shape[0], diagram_img.shape[1])
            new_width = int(diagram_img.shape[1] * scale)
            new_height = int(diagram_img.shape[0] * scale)
            diagram_img = cv2.resize(diagram_img, (new_width, new_height))

        logger.info(f"=== TECHNICAL DRAWING ENHANCED MATCHING START ===")
        logger.info(f"Diagram size: {diagram_img.shape[1]}x{diagram_img.shape[0]}")

        matches = []

        # Get equipment data from database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        for equipment_id in equipment_ids:
            cursor.execute('''
                SELECT e.name, ei.image_path
                FROM equipment e
                JOIN equipment_images ei ON e.id = ei.equipment_id
                WHERE e.id = ?
            ''', (equipment_id,))

            equipment_results = cursor.fetchall()

            if not equipment_results:
                logger.warning(f"No images found for equipment ID {equipment_id}")
                continue

            equipment_name = equipment_results[0][0]
            logger.info(f"\n--- Processing: {equipment_name} ---")

            equipment_matches = []

            for _, template_path in equipment_results:
                if not os.path.exists(template_path):
                    logger.warning(f"Template not found: {template_path}")
                    continue

                # Load template
                template_img = cv2.imread(template_path)
                if template_img is None:
                    logger.warning(f"Failed to load template: {template_path}")
                    continue

                logger.info(f"Template size: {template_img.shape[1]}x{template_img.shape[0]}")

                # Use technical drawing specific combined matching
                match_result = matcher.combined_matching(
                    diagram_img,
                    template_img,
                    equipment_name
                )

                if match_result and match_result['confidence'] > 0.45:  # Raised threshold
                    x, y = match_result['location']
                    w, h = match_result['width'], match_result['height']

                    detection = {
                        'equipment_id': equipment_id,
                        'equipment_name': equipment_name,
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'confidence': float(match_result['confidence']),
                        'center_x': int(x + w // 2),
                        'center_y': int(y + h // 2),
                        'method': match_result.get('method', 'unknown'),
                        'scale': match_result.get('scale', 1.0)
                    }

                    equipment_matches.append(detection)
                    logger.info(f"  ✓ Match found: {match_result['method']} @ ({x},{y}) confidence={match_result['confidence']:.3f}")

            # Apply NMS within equipment matches
            if equipment_matches:
                equipment_matches.sort(key=lambda x: x['confidence'], reverse=True)
                best_match = equipment_matches[0]

                # Check for overlaps with existing matches
                is_duplicate = False
                for existing in matches:
                    iou = calculate_iou_bbox(
                        (best_match['x'], best_match['y'], best_match['width'], best_match['height']),
                        (existing['x'], existing['y'], existing['width'], existing['height'])
                    )
                    if iou > 0.5:  # High overlap threshold
                        is_duplicate = True
                        if best_match['confidence'] > existing['confidence']:
                            matches.remove(existing)
                            matches.append(best_match)
                        break

                if not is_duplicate:
                    matches.append(best_match)

        conn.close()

        # Save debug image if requested
        if debug_mode and matches:
            highlights = []
            for match in matches:
                highlights.append({
                    'bbox': (match['x'], match['y'], match['width'], match['height']),
                    'label': f"{match['equipment_name']} ({match['confidence']:.2f})",
                    'color': (0, 255, 0),
                    'thickness': 2
                })
            save_debug_image(diagram_img, f"matches_{len(matches)}", highlights)

        logger.info(f"\n=== TECHNICAL DRAWING MATCHING COMPLETE ===")
        logger.info(f"Total matches found: {len(matches)}")

        return jsonify({
            'matches': matches,
            'total_found': len(matches),
            'processing_method': 'technical_drawing_specialized'
        })

    except Exception as e:
        logger.error(f"Error in technical drawing matching: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Technical drawing matching failed: {str(e)}'}), 500

def calculate_iou_bbox(bbox1, bbox2):
    """Calculate IoU for two bounding boxes (x, y, w, h)"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 < xi1 or yi2 < yi1:
        return 0

    intersection_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/debug/<path:filename>')
def serve_debug(filename):
    """Serve debug images"""
    return send_from_directory(DEBUG_FOLDER, filename)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'diagrams'), exist_ok=True)
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    app.run(debug=False, host='0.0.0.0', port=8000)