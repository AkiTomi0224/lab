"""
機械学習による機器検出システム
YOLOv8を使用したカスタム物体検出モデルの訓練と推論
"""

import os
import json
import yaml
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import logging

class EquipmentMLTrainer:
    def __init__(self, db_path: str = 'equipment.db'):
        self.db_path = db_path
        self.base_dir = Path('ml_training')
        self.datasets_dir = self.base_dir / 'datasets'
        self.models_dir = self.base_dir / 'models'
        self.temp_dir = self.base_dir / 'temp'

        # ディレクトリ作成
        for dir_path in [self.base_dir, self.datasets_dir, self.models_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)

        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """機械学習用のデータベーステーブルを初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 学習データテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id INTEGER,
            diagram_path TEXT,
            bbox_x INTEGER,
            bbox_y INTEGER,
            bbox_width INTEGER,
            bbox_height INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
        ''')

        # 学習済みモデル管理テーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trained_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id INTEGER,
            model_path TEXT,
            training_samples INTEGER,
            accuracy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
        ''')

        conn.commit()
        conn.close()

    def add_training_sample(self, equipment_id: int, diagram_path: str,
                          bbox: Tuple[int, int, int, int]) -> bool:
        """学習サンプルをデータベースに追加"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            x, y, width, height = bbox
            cursor.execute('''
            INSERT INTO training_data
            (equipment_id, diagram_path, bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (equipment_id, diagram_path, x, y, width, height))

            conn.commit()
            conn.close()

            self.logger.info(f"Training sample added for equipment {equipment_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding training sample: {e}")
            return False

    def get_training_data(self, equipment_id: int) -> List[Dict]:
        """指定した機器の学習データを取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT td.*, e.name as equipment_name
        FROM training_data td
        JOIN equipment e ON td.equipment_id = e.id
        WHERE td.equipment_id = ?
        ORDER BY td.created_at
        ''', (equipment_id,))

        rows = cursor.fetchall()
        conn.close()

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def prepare_yolo_dataset(self, equipment_id: int) -> str:
        """YOLOv8用のデータセット形式を準備"""
        training_data = self.get_training_data(equipment_id)

        if len(training_data) < 5:
            raise ValueError(f"機器ID {equipment_id} の学習データが不足しています（最低5個必要）")

        # データセットディレクトリ作成
        dataset_name = f"equipment_{equipment_id}"
        dataset_dir = self.datasets_dir / dataset_name

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        # YOLOv8のディレクトリ構造を作成
        for split in ['train', 'val']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # データを訓練用と検証用に分割（8:2）
        train_split = int(len(training_data) * 0.8)
        train_data = training_data[:train_split]
        val_data = training_data[train_split:]

        # 各分割でファイルを準備
        for split, data in [('train', train_data), ('val', val_data)]:
            for i, sample in enumerate(data):
                # 画像をコピー
                src_image = sample['diagram_path']
                if not os.path.exists(src_image):
                    self.logger.warning(f"Image not found: {src_image}")
                    continue

                dst_image = dataset_dir / split / 'images' / f"{split}_{i}.jpg"
                shutil.copy2(src_image, dst_image)

                # YOLOフォーマットのラベルファイルを作成
                label_file = dataset_dir / split / 'labels' / f"{split}_{i}.txt"

                # 画像サイズを取得
                img = cv2.imread(src_image)
                img_height, img_width = img.shape[:2]

                # バウンディングボックスを正規化（YOLO形式）
                x_center = (sample['bbox_x'] + sample['bbox_width'] / 2) / img_width
                y_center = (sample['bbox_y'] + sample['bbox_height'] / 2) / img_height
                width = sample['bbox_width'] / img_width
                height = sample['bbox_height'] / img_height

                # ラベルファイル作成（クラスID = 0）
                with open(label_file, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")

        # dataset.yamlファイル作成
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  # クラス数
            'names': [training_data[0]['equipment_name']]
        }

        yaml_file = dataset_dir / 'dataset.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"Dataset prepared: {dataset_dir}")
        return str(yaml_file)

    def train_model(self, equipment_id: int, epochs: int = 100) -> Dict:
        """YOLOv8モデルを訓練"""
        try:
            # データセット準備
            dataset_yaml = self.prepare_yolo_dataset(equipment_id)

            # ベースモデル読み込み（事前訓練済み）
            model = YOLO('yolov8n.pt')  # nanoモデル（軽量）

            # 訓練実行
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=640,
                batch=8,
                device='cpu',  # GPU利用可能な場合は'0'に変更
                project=str(self.models_dir),
                name=f'equipment_{equipment_id}',
                exist_ok=True,
                verbose=True
            )

            # 訓練済みモデルのパス
            model_path = self.models_dir / f'equipment_{equipment_id}' / 'weights' / 'best.pt'

            # データベースに記録
            self._save_trained_model(equipment_id, str(model_path), results)

            return {
                'success': True,
                'model_path': str(model_path),
                'results': results
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _save_trained_model(self, equipment_id: int, model_path: str, results):
        """訓練済みモデル情報をデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 既存のアクティブモデルを無効化
        cursor.execute('''
        UPDATE trained_models SET is_active = FALSE
        WHERE equipment_id = ?
        ''', (equipment_id,))

        # 学習データ数を取得
        cursor.execute('''
        SELECT COUNT(*) FROM training_data WHERE equipment_id = ?
        ''', (equipment_id,))
        sample_count = cursor.fetchone()[0]

        # 新しいモデルを登録
        cursor.execute('''
        INSERT INTO trained_models
        (equipment_id, model_path, training_samples, is_active)
        VALUES (?, ?, ?, TRUE)
        ''', (equipment_id, model_path, sample_count))

        conn.commit()
        conn.close()

    def get_trained_models(self) -> List[Dict]:
        """学習済みモデルの一覧を取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT tm.*, e.name as equipment_name
        FROM trained_models tm
        JOIN equipment e ON tm.equipment_id = e.id
        WHERE tm.is_active = TRUE
        ORDER BY tm.created_at DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def predict(self, equipment_id: int, image_path: str) -> List[Dict]:
        """学習済みモデルで予測"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT model_path FROM trained_models
        WHERE equipment_id = ? AND is_active = TRUE
        ''', (equipment_id,))

        result = cursor.fetchone()
        conn.close()

        if not result:
            return []

        model_path = result[0]

        try:
            # モデル読み込み
            model = YOLO(model_path)

            # 予測実行
            results = model(image_path)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        detections.append({
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(confidence),
                            'equipment_id': equipment_id,
                            'center_x': int((x1 + x2) / 2),
                            'center_y': int((y1 + y2) / 2)
                        })

            return detections

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return []

# インスタンス作成
ml_trainer = EquipmentMLTrainer()
ml_trainer.init_database()