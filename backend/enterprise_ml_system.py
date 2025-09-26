"""
Enterprise-Grade Machine Learning System for Equipment Detection
High-precision batch training data upload and YOLO model management
"""

import os
import json
import sqlite3
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import yaml
from datetime import datetime
import hashlib

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_ml.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TrainingBatch:
    """Enterprise-grade training batch data structure"""
    equipment_name: str
    equipment_id: int
    images: List[str]  # Image file paths
    annotations: List[Dict]  # Annotation data with bounding boxes
    created_at: datetime
    batch_id: str

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = self.generate_batch_id()

    def generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        content = f"{self.equipment_name}_{self.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class MLModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    equipment_id: int
    equipment_name: str
    model_path: str
    training_samples: int
    validation_accuracy: float
    training_time: float
    hyperparameters: Dict
    created_at: datetime
    version: str

class EnterpriseMLSystem:
    """Enterprise-grade ML system with batch processing capabilities"""

    def __init__(self, db_path: str = 'equipment.db'):
        self.db_path = db_path
        self.base_dir = Path('enterprise_ml')
        self.training_data_dir = self.base_dir / 'training_batches'
        self.models_dir = self.base_dir / 'models'
        self.temp_dir = self.base_dir / 'temp'

        # Create directories
        for dir_path in [self.base_dir, self.training_data_dir, self.models_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        self.logger = logging.getLogger(__name__)
        self.init_enterprise_database()

    def init_enterprise_database(self):
        """Initialize enterprise-grade database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Training batches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_batches (
            batch_id TEXT PRIMARY KEY,
            equipment_id INTEGER,
            equipment_name TEXT,
            num_images INTEGER,
            num_annotations INTEGER,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'uploaded',
            metadata TEXT,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
        ''')

        # Enhanced model registry
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            equipment_id INTEGER,
            equipment_name TEXT,
            model_path TEXT,
            training_samples INTEGER,
            validation_accuracy REAL,
            training_time REAL,
            hyperparameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version TEXT,
            status TEXT DEFAULT 'active',
            performance_metrics TEXT,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
        ''')

        # Training annotations with enhanced metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_annotations_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            image_path TEXT,
            bbox_data TEXT,
            image_width INTEGER,
            image_height INTEGER,
            annotation_quality REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES training_batches (batch_id)
        )
        ''')

        conn.commit()
        conn.close()
        self.logger.info("Enterprise database schema initialized")

    def process_training_batch(self, equipment_name: str, batch_files: Dict[str, List]) -> Dict[str, Any]:
        """
        Process batch training data upload

        Args:
            equipment_name: Name of the equipment
            batch_files: Dict containing 'images' and 'annotations' file lists

        Returns:
            Dict with processing results
        """
        try:
            # Validate input
            if not equipment_name or not batch_files.get('images') or not batch_files.get('annotations'):
                raise ValueError("Missing required batch data: equipment_name, images, or annotations")

            # Get or create equipment ID
            equipment_id = self._get_or_create_equipment(equipment_name)

            # Create training batch
            batch = TrainingBatch(
                equipment_name=equipment_name,
                equipment_id=equipment_id,
                images=batch_files['images'],
                annotations=[],
                created_at=datetime.now(),
                batch_id=""
            )

            # Process images and annotations
            processed_data = self._process_batch_data(batch, batch_files)

            # Store in database
            self._store_training_batch(batch, processed_data)

            self.logger.info(f"Successfully processed training batch for {equipment_name}: {len(batch_files['images'])} images")

            return {
                'success': True,
                'batch_id': batch.batch_id,
                'equipment_id': equipment_id,
                'processed_images': len(processed_data['valid_annotations']),
                'message': f'Training batch processed successfully for {equipment_name}'
            }

        except Exception as e:
            self.logger.error(f"Error processing training batch: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_or_create_equipment(self, equipment_name: str) -> int:
        """Get existing equipment ID or create new equipment entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if equipment exists
        cursor.execute('SELECT id FROM equipment WHERE name = ?', (equipment_name,))
        result = cursor.fetchone()

        if result:
            equipment_id = result[0]
        else:
            # Create new equipment entry
            cursor.execute(
                'INSERT INTO equipment (name, created_at) VALUES (?, ?)',
                (equipment_name, datetime.now().isoformat())
            )
            equipment_id = cursor.lastrowid
            self.logger.info(f"Created new equipment entry: {equipment_name} (ID: {equipment_id})")

        conn.commit()
        conn.close()
        return equipment_id

    def _process_batch_data(self, batch: TrainingBatch, batch_files: Dict) -> Dict:
        """Process and validate batch training data"""
        valid_annotations = []
        invalid_files = []

        images = batch_files['images']
        annotations = batch_files.get('annotations', [])

        # Ensure we have annotations for each image
        if len(annotations) != len(images):
            self.logger.warning(f"Mismatch between images ({len(images)}) and annotations ({len(annotations)})")

        for i, (image_path, annotation_data) in enumerate(zip(images, annotations)):
            try:
                # Validate image
                img = cv2.imread(image_path)
                if img is None:
                    invalid_files.append(f"Cannot read image: {image_path}")
                    continue

                height, width = img.shape[:2]

                # Process annotation data
                if isinstance(annotation_data, dict) and 'bbox' in annotation_data:
                    bbox = annotation_data['bbox']

                    # Validate bounding box
                    if self._validate_bbox(bbox, width, height):
                        valid_annotations.append({
                            'image_path': image_path,
                            'bbox': bbox,
                            'width': width,
                            'height': height,
                            'quality': annotation_data.get('quality', 1.0)
                        })
                    else:
                        invalid_files.append(f"Invalid bbox for {image_path}: {bbox}")
                else:
                    invalid_files.append(f"Missing or invalid annotation for {image_path}")

            except Exception as e:
                invalid_files.append(f"Error processing {image_path}: {str(e)}")

        self.logger.info(f"Processed {len(valid_annotations)} valid annotations, {len(invalid_files)} invalid files")

        return {
            'valid_annotations': valid_annotations,
            'invalid_files': invalid_files
        }

    def _validate_bbox(self, bbox: List[float], img_width: int, img_height: int) -> bool:
        """Validate bounding box coordinates"""
        if len(bbox) != 4:
            return False

        x, y, w, h = bbox

        # Check if coordinates are within image bounds
        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
            return False

        # Check if box has valid dimensions
        if w <= 0 or h <= 0:
            return False

        return True

    def _store_training_batch(self, batch: TrainingBatch, processed_data: Dict):
        """Store training batch in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store batch metadata
        metadata = {
            'invalid_files': processed_data['invalid_files'],
            'processing_timestamp': datetime.now().isoformat()
        }

        cursor.execute('''
        INSERT INTO training_batches
        (batch_id, equipment_id, equipment_name, num_images, num_annotations, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            batch.batch_id,
            batch.equipment_id,
            batch.equipment_name,
            len(batch.images),
            len(processed_data['valid_annotations']),
            json.dumps(metadata)
        ))

        # Store individual annotations
        for annotation in processed_data['valid_annotations']:
            cursor.execute('''
            INSERT INTO training_annotations_v2
            (batch_id, image_path, bbox_data, image_width, image_height, annotation_quality)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                batch.batch_id,
                annotation['image_path'],
                json.dumps(annotation['bbox']),
                annotation['width'],
                annotation['height'],
                annotation['quality']
            ))

        conn.commit()
        conn.close()

    def train_enterprise_model(self, equipment_id: int, hyperparameters: Dict = None) -> Dict:
        """
        Train high-precision YOLO model with enterprise-grade configuration
        """
        try:
            start_time = datetime.now()

            # Get training data
            training_data = self._prepare_yolo_dataset_v2(equipment_id)
            if not training_data['success']:
                return training_data

            # Configure hyperparameters
            default_params = {
                'epochs': 300,
                'batch_size': 16,
                'imgsz': 1280,  # Higher resolution for precision
                'patience': 50,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.0005,
                'mosaic': 0.5,  # Data augmentation
                'mixup': 0.15,
                'copy_paste': 0.3,
                'degrees': 10.0,
                'translate': 0.2,
                'scale': 0.9,
                'fliplr': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4
            }

            if hyperparameters:
                default_params.update(hyperparameters)

            # Initialize YOLOv8x model (highest accuracy)
            model = YOLO('yolov8x.pt')  # Use largest model for best accuracy

            # Train with enterprise configuration
            results = model.train(
                data=training_data['dataset_yaml'],
                epochs=default_params['epochs'],
                imgsz=default_params['imgsz'],
                batch=default_params['batch_size'],
                patience=default_params['patience'],
                optimizer=default_params['optimizer'],
                lr0=default_params['lr0'],
                weight_decay=default_params['weight_decay'],
                mosaic=default_params['mosaic'],
                mixup=default_params['mixup'],
                copy_paste=default_params['copy_paste'],
                degrees=default_params['degrees'],
                translate=default_params['translate'],
                scale=default_params['scale'],
                fliplr=default_params['fliplr'],
                hsv_h=default_params['hsv_h'],
                hsv_s=default_params['hsv_s'],
                hsv_v=default_params['hsv_v'],
                device='cpu',  # Change to 'cuda' if GPU available
                project=str(self.models_dir),
                name=f'equipment_{equipment_id}_v2',
                exist_ok=True,
                save_period=50,  # Save checkpoint every 50 epochs
                verbose=True
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # Get validation metrics
            val_accuracy = float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))

            # Save model metadata
            model_metadata = MLModelMetadata(
                model_id=f"model_{equipment_id}_{int(start_time.timestamp())}",
                equipment_id=equipment_id,
                equipment_name=training_data['equipment_name'],
                model_path=str(self.models_dir / f'equipment_{equipment_id}_v2' / 'weights' / 'best.pt'),
                training_samples=training_data['sample_count'],
                validation_accuracy=val_accuracy,
                training_time=training_time,
                hyperparameters=default_params,
                created_at=start_time,
                version="2.0"
            )

            self._store_model_metadata(model_metadata, results)

            self.logger.info(f"Enterprise model training completed for equipment {equipment_id}")
            self.logger.info(f"Training time: {training_time:.2f}s, Validation mAP: {val_accuracy:.4f}")

            return {
                'success': True,
                'model_id': model_metadata.model_id,
                'model_path': model_metadata.model_path,
                'validation_accuracy': val_accuracy,
                'training_time': training_time,
                'sample_count': training_data['sample_count']
            }

        except Exception as e:
            self.logger.error(f"Enterprise model training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _prepare_yolo_dataset_v2(self, equipment_id: int) -> Dict:
        """Prepare enterprise-grade YOLO dataset"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get equipment info
        cursor.execute('SELECT name FROM equipment WHERE id = ?', (equipment_id,))
        equipment_result = cursor.fetchone()
        if not equipment_result:
            return {'success': False, 'error': 'Equipment not found'}

        equipment_name = equipment_result[0]

        # Get all training annotations for this equipment
        cursor.execute('''
        SELECT ta.image_path, ta.bbox_data, ta.image_width, ta.image_height, ta.annotation_quality
        FROM training_annotations_v2 ta
        JOIN training_batches tb ON ta.batch_id = tb.batch_id
        WHERE tb.equipment_id = ?
        ORDER BY ta.created_at
        ''', (equipment_id,))

        annotations = cursor.fetchall()
        conn.close()

        if len(annotations) < 10:  # Require minimum 10 samples for enterprise model
            return {
                'success': False,
                'error': f'Insufficient training data. Need at least 10 samples, found {len(annotations)}'
            }

        # Create dataset directory
        dataset_name = f"enterprise_equipment_{equipment_id}"
        dataset_dir = self.training_data_dir / dataset_name

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Split data (70% train, 20% val, 10% test)
        total_samples = len(annotations)
        train_split = int(total_samples * 0.7)
        val_split = int(total_samples * 0.9)

        splits = {
            'train': annotations[:train_split],
            'val': annotations[train_split:val_split],
            'test': annotations[val_split:]
        }

        # Process each split
        valid_samples = 0
        for split_name, split_data in splits.items():
            for i, (image_path, bbox_json, img_width, img_height, quality) in enumerate(split_data):
                try:
                    # Copy image
                    src_image_path = Path(image_path)
                    if not src_image_path.exists():
                        self.logger.warning(f"Image not found: {image_path}")
                        continue

                    dst_image_path = dataset_dir / split_name / 'images' / f"{split_name}_{i}.jpg"
                    shutil.copy2(src_image_path, dst_image_path)

                    # Create YOLO format label
                    bbox = json.loads(bbox_json)
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height

                    # Write label file
                    label_path = dataset_dir / split_name / 'labels' / f"{split_name}_{i}.txt"
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    valid_samples += 1

                except Exception as e:
                    self.logger.error(f"Error processing sample: {e}")

        # Create dataset.yaml
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': [equipment_name]
        }

        yaml_path = dataset_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"Enterprise dataset prepared: {valid_samples} valid samples")

        return {
            'success': True,
            'dataset_yaml': str(yaml_path),
            'sample_count': valid_samples,
            'equipment_name': equipment_name
        }

    def _store_model_metadata(self, metadata: MLModelMetadata, training_results):
        """Store comprehensive model metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare performance metrics
        performance_metrics = {
            'training_results': str(training_results.results_dict) if hasattr(training_results, 'results_dict') else {},
            'model_size_mb': os.path.getsize(metadata.model_path) / (1024 * 1024) if os.path.exists(metadata.model_path) else 0
        }

        cursor.execute('''
        INSERT INTO model_registry
        (model_id, equipment_id, equipment_name, model_path, training_samples,
         validation_accuracy, training_time, hyperparameters, version, performance_metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.model_id,
            metadata.equipment_id,
            metadata.equipment_name,
            metadata.model_path,
            metadata.training_samples,
            metadata.validation_accuracy,
            metadata.training_time,
            json.dumps(metadata.hyperparameters),
            metadata.version,
            json.dumps(performance_metrics)
        ))

        conn.commit()
        conn.close()

    def get_trained_equipment_list(self) -> List[Dict]:
        """Get list of equipment with trained models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT DISTINCT mr.equipment_id, mr.equipment_name, mr.validation_accuracy,
               mr.training_samples, mr.created_at, mr.model_id
        FROM model_registry mr
        WHERE mr.status = 'active'
        ORDER BY mr.validation_accuracy DESC, mr.created_at DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        trained_equipment = []
        for row in results:
            trained_equipment.append({
                'equipment_id': row[0],
                'equipment_name': row[1],
                'validation_accuracy': row[2],
                'training_samples': row[3],
                'created_at': row[4],
                'model_id': row[5],
                'is_trained': True,
                'accuracy_grade': self._get_accuracy_grade(row[2])
            })

        return trained_equipment

    def _get_accuracy_grade(self, accuracy: float) -> str:
        """Get human-readable accuracy grade"""
        if accuracy >= 0.9:
            return "Excellent (90%+)"
        elif accuracy >= 0.8:
            return "Good (80%+)"
        elif accuracy >= 0.7:
            return "Fair (70%+)"
        else:
            return "Needs Improvement"

    def enterprise_predict(self, equipment_id: int, image_path: str) -> List[Dict]:
        """High-precision prediction using enterprise model"""
        try:
            # Get latest model for equipment
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
            SELECT model_path, model_id, validation_accuracy
            FROM model_registry
            WHERE equipment_id = ? AND status = 'active'
            ORDER BY validation_accuracy DESC, created_at DESC
            LIMIT 1
            ''', (equipment_id,))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return []

            model_path, model_id, accuracy = result

            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return []

            # Load model and predict
            model = YOLO(model_path)

            # Use high confidence threshold for enterprise precision
            results = model(image_path, conf=0.25, iou=0.45, imgsz=1280)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())

                        # Only return high-confidence detections
                        if confidence >= 0.5:
                            detections.append({
                                'x': int(x1),
                                'y': int(y1),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1),
                                'confidence': confidence,
                                'equipment_id': equipment_id,
                                'center_x': int((x1 + x2) / 2),
                                'center_y': int((y1 + y2) / 2),
                                'model_id': model_id,
                                'model_accuracy': accuracy,
                                'detection_method': 'enterprise_ml'
                            })

            self.logger.info(f"Enterprise prediction: {len(detections)} high-confidence detections")
            return detections

        except Exception as e:
            self.logger.error(f"Enterprise prediction failed: {e}")
            return []

# Global enterprise ML system instance
enterprise_ml = EnterpriseMLSystem()