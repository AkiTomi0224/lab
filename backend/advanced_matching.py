"""
Advanced Technical Drawing Analysis and Matching System
高精度図面解析・マッチングシステム
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from skimage import feature, transform, measure
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy.spatial.distance import euclidean, cosine
import math

logger = logging.getLogger(__name__)

class AdvancedTechnicalDrawingMatcher:
    def __init__(self):
        self.debug_mode = True

    def analyze_diagram(self, image: np.ndarray) -> Dict:
        """図面の詳細解析を行う"""
        logger.info("🔍 ADVANCED DIAGRAM ANALYSIS START")

        # 基本情報
        height, width = image.shape[:2]
        logger.info(f"📏 Image dimensions: {width}x{height}")

        # グレースケール変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. エッジ検出による線画解析
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 2. 直線検出 (HoughLinesP)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=30, maxLineGap=10)

        line_info = self._analyze_lines(lines) if lines is not None else {}

        # 3. 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_info = self._analyze_contours(contours)

        # 4. 特徴点検出 (ORB)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # 5. 円・楕円検出
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                  param1=50, param2=30, minRadius=5, maxRadius=100)
        circle_info = self._analyze_circles(circles)

        # 6. テクスチャ解析 (LBP)
        texture_features = self._extract_texture_features(gray)

        analysis_result = {
            'dimensions': (width, height),
            'lines': line_info,
            'contours': contour_info,
            'keypoints': len(keypoints) if keypoints else 0,
            'descriptors': descriptors,
            'circles': circle_info,
            'texture': texture_features,
            'edge_density': np.sum(edges) / (width * height),
            'brightness': np.mean(gray),
            'contrast': np.std(gray)
        }

        logger.info(f"✓ Analysis complete: {len(contours)} contours, {len(keypoints) if keypoints else 0} keypoints, {circle_info['count']} circles")
        return analysis_result

    def _analyze_lines(self, lines: np.ndarray) -> Dict:
        """直線の解析"""
        if lines is None or len(lines) == 0:
            return {'count': 0, 'horizontal': 0, 'vertical': 0, 'diagonal': 0, 'avg_length': 0}

        horizontal = vertical = diagonal = 0
        lengths = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            lengths.append(length)

            # 角度計算
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if angle < 10 or angle > 170:
                horizontal += 1
            elif 80 < angle < 100:
                vertical += 1
            else:
                diagonal += 1

        return {
            'count': len(lines),
            'horizontal': horizontal,
            'vertical': vertical,
            'diagonal': diagonal,
            'avg_length': np.mean(lengths) if lengths else 0
        }

    def _analyze_contours(self, contours: List) -> Dict:
        """輪郭の解析"""
        if not contours:
            return {'count': 0, 'areas': [], 'rectangles': 0, 'complex_shapes': 0}

        areas = []
        rectangles = complex_shapes = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 小さすぎる輪郭は無視
                continue

            areas.append(area)

            # 形状の簡略化
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                rectangles += 1
            elif len(approx) > 8:
                complex_shapes += 1

        return {
            'count': len(contours),
            'areas': areas,
            'rectangles': rectangles,
            'complex_shapes': complex_shapes,
            'avg_area': np.mean(areas) if areas else 0
        }

    def _analyze_circles(self, circles) -> Dict:
        """円の解析"""
        if circles is None:
            return {'count': 0, 'radii': []}

        circles = np.round(circles[0, :]).astype("int")
        radii = [r for (x, y, r) in circles]

        return {
            'count': len(circles),
            'radii': radii,
            'avg_radius': np.mean(radii) if radii else 0
        }

    def _extract_texture_features(self, gray: np.ndarray) -> Dict:
        """テクスチャ特徴抽出"""
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(gray, 24, 3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)

        return {
            'lbp_histogram': lbp_hist,
            'texture_energy': np.sum(lbp_hist ** 2),
            'texture_entropy': -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
        }

    def enhanced_matching(self, diagram_img: np.ndarray, template_img: np.ndarray,
                         equipment_name: str) -> Optional[Dict]:
        """強化されたマッチングアルゴリズム"""
        logger.info(f"🎯 ENHANCED MATCHING: {equipment_name}")

        # 1. 図面とテンプレートの解析
        diagram_analysis = self.analyze_diagram(diagram_img)
        template_analysis = self.analyze_diagram(template_img)

        # 2. 複数スケールでのマッチング
        scales = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
        best_matches = []

        for scale in scales:
            matches = self._multi_method_matching(
                diagram_img, template_img, scale,
                diagram_analysis, template_analysis
            )
            best_matches.extend(matches)

        # 3. マッチング結果の統合と評価
        if best_matches:
            # 非最大値抑制
            final_matches = self._non_max_suppression(best_matches, 0.3)
            if final_matches:
                best_match = max(final_matches, key=lambda x: x['confidence'])
                if best_match['confidence'] > 0.4:  # 閾値調整
                    logger.info(f"✓ ENHANCED MATCH FOUND: {best_match['confidence']:.3f}")
                    return best_match

        logger.info("❌ No reliable match found")
        return None

    def _multi_method_matching(self, diagram_img: np.ndarray, template_img: np.ndarray,
                              scale: float, diagram_analysis: Dict, template_analysis: Dict) -> List[Dict]:
        """複数手法によるマッチング"""
        matches = []

        # テンプレートをスケーリング
        template_height, template_width = template_img.shape[:2]
        new_width = int(template_width * scale)
        new_height = int(template_height * scale)

        if new_width < 10 or new_height < 10:
            return matches

        scaled_template = cv2.resize(template_img, (new_width, new_height))

        # 1. テンプレートマッチング（複数メソッド）
        template_matches = self._template_matching_multi(diagram_img, scaled_template)
        matches.extend(template_matches)

        # 2. 特徴点マッチング
        feature_matches = self._feature_matching(diagram_img, scaled_template)
        matches.extend(feature_matches)

        # 3. 輪郭マッチング
        contour_matches = self._contour_matching(diagram_img, scaled_template)
        matches.extend(contour_matches)

        # 4. 構造的類似度マッチング (SSIM)
        ssim_matches = self._ssim_matching(diagram_img, scaled_template)
        matches.extend(ssim_matches)

        return matches

    def _template_matching_multi(self, diagram_img: np.ndarray, template: np.ndarray) -> List[Dict]:
        """複数メソッドによるテンプレートマッチング"""
        matches = []

        # グレースケール変換
        diagram_gray = cv2.cvtColor(diagram_img, cv2.COLOR_BGR2GRAY) if len(diagram_img.shape) == 3 else diagram_img
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED
        ]

        for method in methods:
            result = cv2.matchTemplate(diagram_gray, template_gray, method)

            if method == cv2.TM_SQDIFF_NORMED:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                confidence = 1 - min_val
                location = min_loc
            else:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                confidence = max_val
                location = max_loc

            if confidence > 0.3:  # 基準値
                matches.append({
                    'location': location,
                    'confidence': confidence,
                    'width': template.shape[1],
                    'height': template.shape[0],
                    'method': f'template_{method}',
                    'center_x': location[0] + template.shape[1] // 2,
                    'center_y': location[1] + template.shape[0] // 2
                })

        return matches

    def _feature_matching(self, diagram_img: np.ndarray, template: np.ndarray) -> List[Dict]:
        """特徴点マッチング"""
        matches = []

        diagram_gray = cv2.cvtColor(diagram_img, cv2.COLOR_BGR2GRAY) if len(diagram_img.shape) == 3 else diagram_img
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        # ORB特徴点検出
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(diagram_gray, None)
        kp2, des2 = orb.detectAndCompute(template_gray, None)

        if des1 is None or des2 is None:
            return matches

        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        feature_matches = bf.match(des1, des2)
        feature_matches = sorted(feature_matches, key=lambda x: x.distance)

        if len(feature_matches) > 10:  # 十分な特徴点マッチが見つかった場合
            # 良いマッチの位置を計算
            good_matches = feature_matches[:min(20, len(feature_matches))]
            points = []
            for match in good_matches:
                img_pt = kp1[match.queryIdx].pt
                points.append(img_pt)

            if points:
                center_x = int(np.mean([p[0] for p in points]))
                center_y = int(np.mean([p[1] for p in points]))

                # 信頼度計算（マッチの質に基づく）
                avg_distance = np.mean([m.distance for m in good_matches])
                confidence = max(0, 1 - (avg_distance / 100))  # 距離を信頼度に変換

                matches.append({
                    'location': (center_x - template.shape[1] // 2, center_y - template.shape[0] // 2),
                    'confidence': confidence,
                    'width': template.shape[1],
                    'height': template.shape[0],
                    'method': 'orb_features',
                    'center_x': center_x,
                    'center_y': center_y,
                    'feature_count': len(good_matches)
                })

        return matches

    def _contour_matching(self, diagram_img: np.ndarray, template: np.ndarray) -> List[Dict]:
        """輪郭マッチング"""
        matches = []

        diagram_gray = cv2.cvtColor(diagram_img, cv2.COLOR_BGR2GRAY) if len(diagram_img.shape) == 3 else diagram_img
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        # エッジ検出
        diagram_edges = cv2.Canny(diagram_gray, 50, 150)
        template_edges = cv2.Canny(template_gray, 50, 150)

        # 輪郭検出
        diagram_contours, _ = cv2.findContours(diagram_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_contours, _ = cv2.findContours(template_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not template_contours:
            return matches

        template_contour = max(template_contours, key=cv2.contourArea)

        for i, diagram_contour in enumerate(diagram_contours):
            if cv2.contourArea(diagram_contour) < 100:  # 小さすぎる輪郭は無視
                continue

            # Hu Moments による形状マッチング
            try:
                match_value = cv2.matchShapes(diagram_contour, template_contour, cv2.CONTOURS_MATCH_I1, 0)
                confidence = max(0, 1 - match_value)  # マッチ値を信頼度に変換

                if confidence > 0.3:
                    # 輪郭の中心計算
                    M = cv2.moments(diagram_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])

                        matches.append({
                            'location': (center_x - template.shape[1] // 2, center_y - template.shape[0] // 2),
                            'confidence': confidence,
                            'width': template.shape[1],
                            'height': template.shape[0],
                            'method': 'contour_matching',
                            'center_x': center_x,
                            'center_y': center_y
                        })
            except:
                continue

        return matches

    def _ssim_matching(self, diagram_img: np.ndarray, template: np.ndarray) -> List[Dict]:
        """構造的類似度マッチング"""
        matches = []

        diagram_gray = cv2.cvtColor(diagram_img, cv2.COLOR_BGR2GRAY) if len(diagram_img.shape) == 3 else diagram_img
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        template_h, template_w = template_gray.shape
        diagram_h, diagram_w = diagram_gray.shape

        if template_h > diagram_h or template_w > diagram_w:
            return matches

        best_ssim = 0
        best_location = (0, 0)

        # スライディングウィンドウでSSIMを計算
        step_size = max(10, min(template_w, template_h) // 4)

        for y in range(0, diagram_h - template_h, step_size):
            for x in range(0, diagram_w - template_w, step_size):
                diagram_patch = diagram_gray[y:y+template_h, x:x+template_w]

                # SSIM計算
                ssim_value = ssim(template_gray, diagram_patch)

                if ssim_value > best_ssim:
                    best_ssim = ssim_value
                    best_location = (x, y)

        if best_ssim > 0.4:  # SSIM閾値
            matches.append({
                'location': best_location,
                'confidence': best_ssim,
                'width': template_w,
                'height': template_h,
                'method': 'ssim',
                'center_x': best_location[0] + template_w // 2,
                'center_y': best_location[1] + template_h // 2
            })

        return matches

    def _non_max_suppression(self, matches: List[Dict], threshold: float) -> List[Dict]:
        """非最大値抑制"""
        if not matches:
            return []

        # 信頼度でソート
        matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

        keep = []

        for match in matches:
            overlap = False
            for kept_match in keep:
                # IoU計算
                iou = self._calculate_iou(match, kept_match)
                if iou > threshold:
                    overlap = True
                    break

            if not overlap:
                keep.append(match)

        return keep

    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """IoU (Intersection over Union) 計算"""
        x1_1, y1_1 = box1['location']
        x2_1 = x1_1 + box1['width']
        y2_1 = y1_1 + box1['height']

        x1_2, y1_2 = box2['location']
        x2_2 = x1_2 + box2['width']
        y2_2 = y1_2 + box2['height']

        # 交差領域
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0