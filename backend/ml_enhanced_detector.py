"""
Machine Learning Enhanced Detection System
Combines traditional computer vision with ML features for equipment detection
"""

import cv2
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sqlite3
from typing import List, Dict, Tuple, Optional
import json
import time

logger = logging.getLogger(__name__)

class MLEnhancedDetector:
    """ML-Enhanced detector for technical drawings"""

    def __init__(self, database_path: str):
        # Use the correct database file
        if database_path.endswith('equipment_detection.db'):
            self.database_path = database_path.replace('equipment_detection.db', 'equipment.db')
        else:
            self.database_path = database_path
        self.feature_extractor = FeatureExtractor()
        self.ml_classifier = None
        self.scaler = StandardScaler()
        self.trained_features = {}
        self.model_cache = {}

        # Load any existing trained models
        self.load_trained_models()

    def train_equipment_classifier(self, equipment_id: int, equipment_name: str) -> bool:
        """Train ML classifier for specific equipment using training data"""
        try:
            logger.info(f"🧠 Training ML classifier for {equipment_name} (ID: {equipment_id})")

            # Get training data
            training_data = self.get_training_data(equipment_id)
            if len(training_data) < 2:
                logger.warning(f"Insufficient training data for {equipment_name} ({len(training_data)} samples)")
                return False

            # Extract features from training data
            positive_features = []
            for image_path, highlight_path in training_data:
                features = self.extract_ml_features(image_path, highlight_path)
                if features is not None:
                    positive_features.append(features)

            if len(positive_features) < 2:
                logger.warning(f"Could not extract sufficient features for {equipment_name}")
                return False

            # Generate negative samples (from other equipment)
            negative_features = self.generate_negative_samples(equipment_id, len(positive_features))

            # Prepare training data
            X = np.array(positive_features + negative_features)
            y = np.array([1] * len(positive_features) + [0] * len(negative_features))

            # Train classifier
            classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train
            classifier.fit(X_scaled, y)

            # Store trained model
            self.model_cache[equipment_id] = {
                'classifier': classifier,
                'scaler': scaler,
                'feature_template': np.mean(positive_features, axis=0)
            }

            # Save to disk
            self.save_trained_model(equipment_id, equipment_name)

            logger.info(f"✅ Successfully trained classifier for {equipment_name}")
            return True

        except Exception as e:
            logger.error(f"Training failed for {equipment_name}: {str(e)}")
            return False

    def detect_with_ml(self, diagram_image: np.ndarray, equipment_id: int, equipment_name: str) -> Optional[Dict]:
        """Perform ML-enhanced detection"""
        try:
            logger.info(f"🔍 ML Detection for {equipment_name}")

            # Check if we have a trained model
            if equipment_id not in self.model_cache:
                logger.warning(f"No trained model for {equipment_name}")
                return None

            model_data = self.model_cache[equipment_id]
            classifier = model_data['classifier']
            scaler = model_data['scaler']
            template_features = model_data['feature_template']

            # Sliding window detection with ML
            best_detection = None
            best_confidence = 0

            h, w = diagram_image.shape[:2]
            logger.info(f"🖼️ Processing image size: {w}x{h}")

            # Optimized scales and window sizes for better performance
            scales = [0.7, 1.0, 1.3]  # Reduced from 5 to 3 scales
            window_sizes = [(80, 80), (120, 120), (150, 150)]  # Reduced window sizes

            total_windows = 0
            processed_windows = 0
            start_time = time.time()
            timeout_seconds = 30  # Maximum 30 seconds for detection

            for scale_idx, scale in enumerate(scales):
                logger.info(f"🔍 Processing scale {scale_idx+1}/{len(scales)}: {scale}")

                for size_idx, (base_w, base_h) in enumerate(window_sizes):
                    window_w = int(base_w * scale)
                    window_h = int(base_h * scale)

                    if window_w >= w or window_h >= h:
                        continue

                    # Larger step size for better performance
                    step_size = max(window_w // 2, 40)  # Increased step size

                    # Calculate number of positions
                    x_positions = range(0, w - window_w, step_size)
                    y_positions = range(0, h - window_h, step_size)
                    window_count = len(x_positions) * len(y_positions)
                    total_windows += window_count

                    logger.info(f"   📐 Window size {window_w}x{window_h}: {window_count} positions")

                    for y_idx, y in enumerate(y_positions):
                        for x_idx, x in enumerate(x_positions):
                            # Check for timeout
                            elapsed_time = time.time() - start_time
                            if elapsed_time > timeout_seconds:
                                logger.warning(f"⏰ Detection timeout after {elapsed_time:.1f} seconds")
                                break

                            processed_windows += 1

                            # Progress logging every 100 windows
                            if processed_windows % 100 == 0:
                                progress = (processed_windows / total_windows) * 100
                                logger.info(f"   🔄 Progress: {processed_windows}/{total_windows} ({progress:.1f}%) - {elapsed_time:.1f}s")

                            window = diagram_image[y:y+window_h, x:x+window_w]

                            # Extract features for this window
                            window_features = self.feature_extractor.extract_region_features(window)
                            if window_features is None:
                                continue

                            # Scale features
                            window_features_scaled = scaler.transform([window_features])

                            # ML prediction
                            ml_prob = classifier.predict_proba(window_features_scaled)[0][1]

                            # Template similarity
                            template_sim = cosine_similarity([window_features], [template_features])[0][0]

                            # Combined confidence
                            confidence = (ml_prob * 0.7 + template_sim * 0.3)

                            if confidence > best_confidence and confidence > 0.3:
                                best_confidence = confidence
                                best_detection = {
                                    'location': (x, y),
                                    'width': window_w,
                                    'height': window_h,
                                    'confidence': confidence,
                                    'method': 'ML-Enhanced',
                                    'ml_probability': ml_prob,
                                    'template_similarity': template_sim,
                                    'scale': scale
                                }
                                logger.info(f"   🎯 New best detection: {confidence:.3f} at ({x},{y})")

                        # Break outer loop if timeout occurred
                        if elapsed_time > timeout_seconds:
                            break

                    # Break outer loop if timeout occurred
                    if elapsed_time > timeout_seconds:
                        break

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Processed {processed_windows} windows total in {elapsed_time:.1f} seconds")

            if best_detection:
                logger.info(f"✅ ML Detection success: conf={best_confidence:.3f}")
            else:
                logger.info(f"❌ ML Detection failed for {equipment_name}")

            return best_detection

        except Exception as e:
            logger.error(f"ML detection error: {str(e)}")
            return None

    def extract_ml_features(self, image_path: str, highlight_path: str = None) -> Optional[np.ndarray]:
        """Extract ML features from training image"""
        try:
            # Complete path resolution for training images
            full_paths_to_try = []

            if os.path.isabs(image_path):
                full_paths_to_try.append(image_path)
            else:
                # Handle relative paths properly
                # The database stores paths like ../uploads/training_sets_xxx/...
                # We need to resolve these correctly from the backend directory

                # Method 1: Direct resolution from backend directory
                backend_resolved = os.path.join(os.path.dirname(__file__), image_path)
                full_paths_to_try.append(os.path.normpath(backend_resolved))

                # Method 2: Project root + strip .. prefix
                if image_path.startswith('../'):
                    clean_path = image_path[3:]  # Remove '../' prefix
                    project_root_path = os.path.join('/Users/tomitaakihiro/図面可視化システム', clean_path)
                    full_paths_to_try.append(project_root_path)

                # Method 3: Direct uploads folder resolution
                if 'uploads/' in image_path:
                    uploads_part = image_path[image_path.find('uploads/'):]
                    uploads_path = os.path.join('/Users/tomitaakihiro/図面可視化システム', uploads_part)
                    full_paths_to_try.append(uploads_path)

            image = None
            successful_path = None

            for full_path in full_paths_to_try:
                try:
                    if os.path.exists(full_path):
                        image = cv2.imread(full_path)
                        if image is not None:
                            successful_path = full_path
                            logger.info(f"✅ Successfully loaded image from: {successful_path}")
                            break
                except Exception as e:
                    logger.debug(f"Failed to load from {full_path}: {e}")
                    continue

            if image is None:
                logger.error(f"❌ Could not load image from any of these paths: {full_paths_to_try}")
                return None
            return self.feature_extractor.extract_comprehensive_features(image)

        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None

    def get_training_data(self, equipment_id: int) -> List[Tuple[str, str]]:
        """Get training data for equipment"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Query from the correct table structure
            cursor.execute("""
                SELECT ta.image_path, ta.image_path
                FROM training_annotations_v2 ta
                JOIN training_batches tb ON ta.batch_id = tb.batch_id
                WHERE tb.equipment_id = ?
                ORDER BY ta.created_at DESC
            """, (equipment_id,))
            results = cursor.fetchall()

            conn.close()

            return [(row[0], row[1]) for row in results if row[0] and row[1]]

        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return []

    def generate_negative_samples(self, target_equipment_id: int, num_samples: int) -> List[np.ndarray]:
        """Generate negative samples from other equipment"""
        try:
            negative_features = []

            # Get other equipment training data
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT ta.image_path
                FROM training_annotations_v2 ta
                JOIN training_batches tb ON ta.batch_id = tb.batch_id
                WHERE tb.equipment_id != ?
                ORDER BY RANDOM()
                LIMIT ?
            """, (target_equipment_id, num_samples))

            results = cursor.fetchall()
            conn.close()

            for (image_path,) in results:
                features = self.extract_ml_features(image_path)
                if features is not None:
                    negative_features.append(features)

            # If not enough negative samples, generate synthetic ones
            while len(negative_features) < num_samples:
                synthetic_features = np.random.normal(0, 0.5, size=30).astype(np.float32)  # Match actual feature size
                negative_features.append(synthetic_features)

            return negative_features[:num_samples]

        except Exception as e:
            logger.error(f"Error generating negative samples: {str(e)}")
            return []

    def save_trained_model(self, equipment_id: int, equipment_name: str):
        """Save trained model to disk"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
            os.makedirs(model_dir, exist_ok=True)

            model_file = os.path.join(model_dir, f'equipment_{equipment_id}_{equipment_name}.pkl')

            with open(model_file, 'wb') as f:
                pickle.dump(self.model_cache[equipment_id], f)

            logger.info(f"Model saved: {model_file}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_trained_models(self):
        """Load existing trained models from disk"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
            if not os.path.exists(model_dir):
                return

            for filename in os.listdir(model_dir):
                if filename.endswith('.pkl') and filename.startswith('equipment_'):
                    # Extract equipment ID from filename
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        try:
                            equipment_id = int(parts[1])
                            model_file = os.path.join(model_dir, filename)

                            with open(model_file, 'rb') as f:
                                self.model_cache[equipment_id] = pickle.load(f)

                            logger.info(f"Loaded model for equipment {equipment_id}")

                        except (ValueError, Exception) as e:
                            logger.warning(f"Could not load model {filename}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")


class FeatureExtractor:
    """Advanced feature extraction for technical drawings"""

    def __init__(self):
        # Initialize feature detectors
        self.sift = cv2.SIFT_create(nfeatures=100)
        self.orb = cv2.ORB_create(nfeatures=100)

    def extract_comprehensive_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive features for ML"""
        try:
            # Prepare image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Feature categories - ensure consistent dimensions
            features = []

            # 1. Shape features (10 features)
            shape_features = self.extract_shape_features(gray)
            features.extend(shape_features)

            # 2. Texture features (10 features)
            texture_features = self.extract_texture_features(gray)
            features.extend(texture_features)

            # 3. Edge features (5 features)
            edge_features = self.extract_edge_features(gray)
            features.extend(edge_features)

            # 4. Geometric features (5 features)
            geometric_features = self.extract_geometric_features(gray)
            features.extend(geometric_features)

            # Ensure exactly 30 features
            features = features[:30]
            while len(features) < 30:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return np.zeros(30, dtype=np.float32)

    def extract_region_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for sliding window region"""
        # Resize to standard size for consistent features
        standardized = cv2.resize(image, (100, 100))
        return self.extract_comprehensive_features(standardized)

    def extract_shape_features(self, gray: np.ndarray) -> List[float]:
        """Extract shape-based features"""
        features = []

        # Contour-based features
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Largest contour features
            largest_contour = max(contours, key=cv2.contourArea)

            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            features.extend([
                area / (gray.shape[0] * gray.shape[1]),  # Normalized area
                perimeter / (2 * (gray.shape[0] + gray.shape[1])),  # Normalized perimeter
                4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,  # Circularity
            ])

            # Hu moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend([-np.sign(hu) * np.log10(np.abs(hu) + 1e-10) for hu in hu_moments])
        else:
            features.extend([0] * 10)  # Default values

        return features[:10]  # Ensure consistent length

    def extract_texture_features(self, gray: np.ndarray) -> List[float]:
        """Extract texture features using LBP and statistics"""
        features = []

        # Basic statistics
        features.extend([
            np.mean(gray) / 255.0,
            np.std(gray) / 255.0,
            np.min(gray) / 255.0,
            np.max(gray) / 255.0
        ])

        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.extend([
            np.mean(gradient_magnitude) / 255.0,
            np.std(gradient_magnitude) / 255.0
        ])

        return features[:10]  # Limit to 10 features

    def extract_edge_features(self, gray: np.ndarray) -> List[float]:
        """Extract edge-based features"""
        features = []

        # Canny edges
        edges = cv2.Canny(gray, 50, 150)

        # Edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)

        if lines is not None:
            # Line statistics
            line_lengths = []
            line_angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1)

                line_lengths.append(length)
                line_angles.append(angle)

            features.extend([
                len(lines) / 100.0,  # Normalized number of lines
                np.mean(line_lengths) if line_lengths else 0,
                np.std(line_angles) if line_angles else 0
            ])
        else:
            features.extend([0, 0, 0])

        return features[:5]  # Limit to 5 features

    def extract_geometric_features(self, gray: np.ndarray) -> List[float]:
        """Extract geometric features"""
        features = []

        # Aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 1
        features.append(aspect_ratio)

        # Solidity and extent (requires contours)
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            solidity = contour_area / hull_area if hull_area > 0 else 0

            # Extent
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h
            extent = contour_area / rect_area if rect_area > 0 else 0

            features.extend([solidity, extent])
        else:
            features.extend([0, 0])

        return features[:5]  # Limit to 5 features