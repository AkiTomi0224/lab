#!/usr/bin/env python3

"""
Auto-cleanup module for removing orphaned training data entries.
This module is automatically called whenever training data is deleted.
"""

import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    db_path = '/Users/tomitaakihiro/図面可視化システム/backend/equipment.db'
    return sqlite3.connect(db_path)

def auto_cleanup_orphaned_data():
    """
    Automatically remove training data entries that reference non-existent files.
    Returns the number of orphaned records removed.
    """

    logger.info("🧹 Auto-cleanup: Starting orphaned data cleanup...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check training_annotations_v2 for missing image files
    cursor.execute("SELECT id, image_path FROM training_annotations_v2")
    annotations = cursor.fetchall()

    orphaned_count = 0
    valid_count = 0

    for ann_id, img_path in annotations:
        if img_path:
            # Try multiple possible paths
            possible_paths = [
                img_path,
                os.path.join('/Users/tomitaakihiro/図面可視化システム', img_path),
                os.path.join('/Users/tomitaakihiro/図面可視化システム', img_path.lstrip('/'))
            ]

            path_exists = False
            for path in possible_paths:
                if os.path.exists(path):
                    path_exists = True
                    break

            if not path_exists:
                logger.info(f"🗑️  Auto-cleanup: Removing orphaned annotation {ann_id}: {img_path}")
                cursor.execute("DELETE FROM training_annotations_v2 WHERE id = ?", (ann_id,))
                orphaned_count += 1
            else:
                valid_count += 1

    # Check training_batches table for empty batches
    cursor.execute("SELECT batch_id, equipment_id, equipment_name FROM training_batches")
    batches = cursor.fetchall()

    batch_orphaned_count = 0
    batch_valid_count = 0

    for batch_id, eq_id, eq_name in batches:
        # Check if this batch has any valid training annotations
        cursor.execute("SELECT COUNT(*) FROM training_annotations_v2 WHERE batch_id = ?", (batch_id,))
        annotation_count = cursor.fetchone()[0]

        if annotation_count == 0:
            logger.info(f"🗑️  Auto-cleanup: Removing empty training batch {batch_id} for {eq_name}")
            cursor.execute("DELETE FROM training_batches WHERE batch_id = ?", (batch_id,))
            batch_orphaned_count += 1
        else:
            batch_valid_count += 1

    # Commit changes
    conn.commit()
    conn.close()

    total_removed = orphaned_count + batch_orphaned_count

    if total_removed > 0:
        logger.info(f"✅ Auto-cleanup completed: {orphaned_count} annotations + {batch_orphaned_count} batches = {total_removed} orphaned records removed")
    else:
        logger.info("✨ Auto-cleanup completed: No orphaned data found")

    return total_removed

def auto_cleanup_trained_models():
    """
    Remove trained model files for equipment that no longer has training data.
    """

    logger.info("🤖 Auto-cleanup: Cleaning up orphaned trained models...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all equipment IDs that have training data
    cursor.execute("""
        SELECT DISTINCT equipment_id
        FROM training_batches
        WHERE batch_id IN (
            SELECT DISTINCT batch_id
            FROM training_annotations_v2
        )
    """)

    valid_equipment_ids = set(row[0] for row in cursor.fetchall())
    conn.close()

    # Check trained models directory
    models_dir = '/Users/tomitaakihiro/図面可視化システム/backend/trained_models'
    removed_models = 0

    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl'):
                # Extract equipment ID from filename (format: equipment_X_name.pkl)
                try:
                    parts = filename.split('_')
                    if len(parts) >= 2 and parts[0] == 'equipment':
                        equipment_id = int(parts[1])

                        if equipment_id not in valid_equipment_ids:
                            model_path = os.path.join(models_dir, filename)
                            os.remove(model_path)
                            logger.info(f"🤖 Auto-cleanup: Removed orphaned model {filename}")
                            removed_models += 1
                except (ValueError, IndexError):
                    # Skip files that don't match expected format
                    continue

    if removed_models > 0:
        logger.info(f"✅ Auto-cleanup: Removed {removed_models} orphaned trained models")
    else:
        logger.info("✨ Auto-cleanup: No orphaned trained models found")

    return removed_models

def full_auto_cleanup():
    """
    Perform complete automatic cleanup of orphaned data and models.
    Returns total number of items cleaned up.
    """

    orphaned_records = auto_cleanup_orphaned_data()
    orphaned_models = auto_cleanup_trained_models()

    total_cleaned = orphaned_records + orphaned_models

    if total_cleaned > 0:
        logger.info(f"🎯 Full auto-cleanup completed: {total_cleaned} items cleaned up")
    else:
        logger.info("🎯 Full auto-cleanup completed: System is already clean")

    return total_cleaned

if __name__ == "__main__":
    # Manual execution for testing
    full_auto_cleanup()