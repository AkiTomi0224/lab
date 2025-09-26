#!/usr/bin/env python3

"""
Database cleanup script to remove orphaned training data entries
that reference non-existent image files.

This addresses the "Could not load image" errors by cleaning up
database records that point to deleted or moved files.
"""

import os
import sqlite3
import sys

def get_db_connection():
    """Get database connection"""
    db_path = '/Users/tomitaakihiro/図面可視化システム/backend/equipment.db'
    return sqlite3.connect(db_path)

def cleanup_orphaned_training_data():
    """Remove training data entries that reference non-existent files"""

    print("🧹 Starting database cleanup...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Start with training_annotations_v2 since this contains the actual image paths
    print("\n📋 Checking training_annotations_v2 table for missing image files...")
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
                print(f"❌ Orphaned annotation {ann_id}: {img_path}")
                print(f"   - Tried paths: {possible_paths}")
                cursor.execute("DELETE FROM training_annotations_v2 WHERE id = ?", (ann_id,))
                orphaned_count += 1
            else:
                valid_count += 1

    # Check training_batches table
    print("\n📋 Checking training_batches table...")
    cursor.execute("SELECT batch_id, equipment_id, equipment_name FROM training_batches")
    batches = cursor.fetchall()

    batch_orphaned_count = 0
    batch_valid_count = 0

    for batch_id, eq_id, eq_name in batches:
        # Check if this batch has any valid training annotations
        cursor.execute("SELECT COUNT(*) FROM training_annotations_v2 WHERE batch_id = ?", (batch_id,))
        annotation_count = cursor.fetchone()[0]

        if annotation_count == 0:
            print(f"❌ Empty training batch {batch_id} for {eq_name} (equipment_id: {eq_id})")
            cursor.execute("DELETE FROM training_batches WHERE batch_id = ?", (batch_id,))
            batch_orphaned_count += 1
        else:
            batch_valid_count += 1

    # Commit changes
    conn.commit()
    conn.close()

    # Summary
    print(f"\n✅ Cleanup completed:")
    print(f"   📝 Annotations: {orphaned_count} deleted, {valid_count} valid")
    print(f"   📦 Training batches: {batch_orphaned_count} deleted, {batch_valid_count} valid")
    print(f"   🎯 Total orphaned records removed: {orphaned_count + batch_orphaned_count}")

    return orphaned_count + batch_orphaned_count > 0

def list_existing_training_data():
    """List all valid training data that remains after cleanup"""

    print("\n📊 Remaining valid training data:")

    conn = get_db_connection()
    cursor = conn.cursor()

    # List valid annotations
    cursor.execute("""
        SELECT ta.id, ta.image_path, tb.equipment_name, tb.equipment_id
        FROM training_annotations_v2 ta
        JOIN training_batches tb ON ta.batch_id = tb.batch_id
        ORDER BY tb.equipment_id, ta.id
        LIMIT 10
    """)

    annotations = cursor.fetchall()

    if not annotations:
        print("   ⚠️  No valid annotations found")
    else:
        print("   📝 Sample annotations (first 10):")
        for ann_id, img_path, eq_name, eq_id in annotations:
            print(f"      📷 {eq_name}: {os.path.basename(img_path) if img_path else 'N/A'}")

    # List valid training batches
    cursor.execute("""
        SELECT tb.batch_id, tb.equipment_id, tb.equipment_name, COUNT(ta.id) as annotation_count
        FROM training_batches tb
        LEFT JOIN training_annotations_v2 ta ON tb.batch_id = ta.batch_id
        GROUP BY tb.batch_id, tb.equipment_id, tb.equipment_name
        ORDER BY tb.equipment_id
    """)

    batches = cursor.fetchall()

    if batches:
        print(f"\n   📦 Training batches:")
        for batch_id, eq_id, eq_name, count in batches:
            print(f"      🎯 Batch {batch_id}: {eq_name} ({count} annotations)")

    conn.close()

if __name__ == "__main__":
    print("🔥 Database Cleanup Tool")
    print("=" * 50)

    # Perform cleanup
    changes_made = cleanup_orphaned_training_data()

    # List remaining data
    list_existing_training_data()

    if changes_made:
        print(f"\n🚀 Database cleanup completed! Please restart the Flask server to see improvements.")
        print(f"   The 'Could not load image' errors should be significantly reduced.")
    else:
        print(f"\n✨ No orphaned data found - database is already clean!")

    print(f"\n💡 Next steps:")
    print(f"   1. Restart your Flask server")
    print(f"   2. Test the detection system")
    print(f"   3. Check that ML training only runs when needed")