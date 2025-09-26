#!/usr/bin/env python3

"""
Deep cleanup script to completely remove all orphaned training data
and reset the system to a clean state.
"""

import os
import sqlite3
import shutil

def get_db_connection():
    """Get database connection"""
    db_path = '/Users/tomitaakihiro/図面可視化システム/backend/equipment.db'
    return sqlite3.connect(db_path)

def deep_cleanup():
    """Perform a deep cleanup of all training data"""

    print("🔥 Deep System Cleanup")
    print("=" * 50)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Delete all training-related data
    print("🧹 Cleaning database tables...")

    tables_to_clean = [
        'training_annotations_v2',
        'training_batches',
        'training_sets',
        'equipment_images'
    ]

    for table in tables_to_clean:
        try:
            cursor.execute(f"DELETE FROM {table}")
            count = cursor.rowcount
            print(f"   📋 {table}: {count} records deleted")
        except sqlite3.OperationalError as e:
            print(f"   ⚠️  {table}: Table not found or error - {e}")

    # Keep equipment definitions but reset status
    cursor.execute("UPDATE equipment SET status = 'untrained' WHERE status = 'trained'")
    updated_count = cursor.rowcount
    print(f"   🔧 equipment: Reset {updated_count} equipment to 'untrained' status")

    conn.commit()
    conn.close()

    # Clean up file system
    print("\n🧹 Cleaning file system...")

    base_path = '/Users/tomitaakihiro/図面可視化システム'

    # Remove all training sets
    uploads_path = os.path.join(base_path, 'uploads')
    if os.path.exists(uploads_path):
        for item in os.listdir(uploads_path):
            if item.startswith('training_sets_'):
                full_path = os.path.join(uploads_path, item)
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                    print(f"   🗂️  Removed directory: {item}")

    # Remove all trained models
    models_path = os.path.join(base_path, 'backend', 'trained_models')
    if os.path.exists(models_path):
        for item in os.listdir(models_path):
            if item.endswith('.pkl'):
                full_path = os.path.join(models_path, item)
                os.remove(full_path)
                print(f"   🤖 Removed model: {item}")

    print("\n✅ Deep cleanup completed!")
    print("   🔄 System reset to clean state")
    print("   📝 All training data removed")
    print("   🎯 No more orphaned file references")

    print(f"\n💡 Next steps:")
    print(f"   1. Restart your Flask server")
    print(f"   2. Upload fresh diagrams")
    print(f"   3. Create new training sets")
    print(f"   4. Test the detection system")

if __name__ == "__main__":
    deep_cleanup()