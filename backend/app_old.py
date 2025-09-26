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

app = Flask(__name__, static_folder=None)  # Disable default static folder
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'equipment.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Create equipment table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create equipment images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipment_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id INTEGER,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (equipment_id) REFERENCES equipment (id)
        )
    ''')

    # Create diagrams table
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

# High-precision image matching functions
def preprocess_image_for_matching(img):
    """Advanced preprocessing for technical drawings"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Noise reduction with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Histogram equalization for better contrast
    equalized = cv2.equalizeHist(denoised)

    # Gaussian blur to reduce fine noise
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    return blurred

def multiscale_template_matching(diagram, template, scales=[0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]):
    """Template matching at multiple scales"""
    best_match = None
    best_confidence = 0

    diagram_preprocessed = preprocess_image_for_matching(diagram)
    template_preprocessed = preprocess_image_for_matching(template)

    for scale in scales:
        # Skip if template becomes too large
        new_width = int(template_preprocessed.shape[1] * scale)
        new_height = int(template_preprocessed.shape[0] * scale)

        if new_width >= diagram_preprocessed.shape[1] or new_height >= diagram_preprocessed.shape[0]:
            continue

        if new_width < 10 or new_height < 10:
            continue

        # Resize template
        scaled_template = cv2.resize(template_preprocessed, (new_width, new_height))

        # Multiple template matching methods
        methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED
        ]

        for method in methods:
            result = cv2.matchTemplate(diagram_preprocessed, scaled_template, method)

            if method == cv2.TM_SQDIFF_NORMED:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                confidence = 1 - min_val  # Convert to similarity score
                location = min_loc
            else:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                confidence = max_val
                location = max_loc

            # Adjust confidence threshold based on method and scale
            threshold = 0.3 if method == cv2.TM_SQDIFF_NORMED else 0.4

            if confidence > threshold and confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    'location': location,
                    'scale': scale,
                    'method': method,
                    'confidence': confidence,
                    'width': new_width,
                    'height': new_height
                }

    return best_match

def feature_based_matching(diagram, template):
    """SIFT/ORB feature-based matching for rotation/scale invariance"""
    try:
        diagram_prep = preprocess_image_for_matching(diagram)
        template_prep = preprocess_image_for_matching(template)

        # Try SIFT first (more accurate but slower)
        try:
            sift = cv2.SIFT_create(nfeatures=1000)
            kp1, des1 = sift.detectAndCompute(template_prep, None)
            kp2, des2 = sift.detectAndCompute(diagram_prep, None)

            if des1 is not None and des2 is not None and len(des1) > 4 and len(des2) > 4:
                # FLANN matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= 4:
                    # Extract location of good matches
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Find homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        # Get corners of template in diagram
                        h, w = template_prep.shape
                        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # Calculate center and confidence
                        center_x = np.mean(dst[:, 0, 0])
                        center_y = np.mean(dst[:, 0, 1])

                        # Confidence based on inliers ratio
                        inliers = np.sum(mask)
                        confidence = min(inliers / len(good_matches), 1.0)

                        if confidence > 0.3:  # At least 30% inliers
                            return {
                                'location': (int(center_x - w/2), int(center_y - h/2)),
                                'confidence': confidence,
                                'width': w,
                                'height': h,
                                'method': 'SIFT'
                            }
        except:
            pass

        # Fallback to ORB (faster but less accurate)
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(template_prep, None)
        kp2, des2 = orb.detectAndCompute(diagram_prep, None)

        if des1 is not None and des2 is not None:
            # BFMatcher with hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) >= 4:
                # Extract location of good matches (top 50%)
                good_matches = matches[:max(4, len(matches)//2)]
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h, w = template_prep.shape
                    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    center_x = np.mean(dst[:, 0, 0])
                    center_y = np.mean(dst[:, 0, 1])

                    inliers = np.sum(mask) if mask is not None else len(good_matches)
                    confidence = min(inliers / len(good_matches), 1.0) * 0.8  # ORB penalty

                    if confidence > 0.25:
                        return {
                            'location': (int(center_x - w/2), int(center_y - h/2)),
                            'confidence': confidence,
                            'width': w,
                            'height': h,
                            'method': 'ORB'
                        }
    except:
        pass

    return None

def edge_based_matching(diagram, template):
    """Edge-based matching for line drawings"""
    try:
        # Edge detection
        diagram_edges = cv2.Canny(preprocess_image_for_matching(diagram), 50, 150)
        template_edges = cv2.Canny(preprocess_image_for_matching(template), 50, 150)

        # Dilate edges to make them thicker
        kernel = np.ones((3, 3), np.uint8)
        diagram_edges = cv2.dilate(diagram_edges, kernel, iterations=1)
        template_edges = cv2.dilate(template_edges, kernel, iterations=1)

        # Template matching on edges
        result = cv2.matchTemplate(diagram_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.3:  # Lower threshold for edge matching
            h, w = template_edges.shape
            return {
                'location': max_loc,
                'confidence': max_val * 0.9,  # Edge matching penalty
                'width': w,
                'height': h,
                'method': 'EDGE'
            }
    except:
        pass

    return None

def combine_matching_results(results):
    """Combine results from multiple matching methods"""
    if not results:
        return None

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return None

    # Weighted voting based on confidence and method reliability
    method_weights = {
        'SIFT': 1.0,
        'ORB': 0.8,
        'EDGE': 0.6,
        cv2.TM_CCOEFF_NORMED: 0.7,
        cv2.TM_CCORR_NORMED: 0.6,
        cv2.TM_SQDIFF_NORMED: 0.5
    }

    weighted_results = []
    for result in valid_results:
        method = result.get('method', 'UNKNOWN')
        weight = method_weights.get(method, 0.5)
        weighted_confidence = result['confidence'] * weight

        weighted_results.append({
            **result,
            'weighted_confidence': weighted_confidence
        })

    # Sort by weighted confidence
    weighted_results.sort(key=lambda x: x['weighted_confidence'], reverse=True)

    # Return the best result if it meets minimum threshold
    best_result = weighted_results[0]
    if best_result['weighted_confidence'] > 0.2:
        return best_result

    return None

def segment_diagram_regions(diagram_img):
    """Segment diagram into individual equipment regions using advanced CV techniques"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(diagram_img, cv2.COLOR_BGR2GRAY)

        # Advanced preprocessing
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. Adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 3. Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 4. Find contours (potential equipment regions)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Filter contours by size and aspect ratio
        segments = []
        min_area = 500  # Minimum area for equipment
        max_area = gray.shape[0] * gray.shape[1] * 0.3  # Max 30% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by aspect ratio (avoid too thin/wide regions)
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio <= 5:  # Not too elongated
                    # Extract region
                    region = gray[y:y+h, x:x+w]
                    segments.append({
                        'region': region,
                        'bbox': (x, y, w, h),
                        'area': area
                    })

        # Sort by area (largest first)
        segments.sort(key=lambda x: x['area'], reverse=True)

        print(f"Found {len(segments)} potential equipment regions")

        return segments

    except Exception as e:
        print(f"Error in diagram segmentation: {e}")
        return []

def match_segment_with_equipment(segment_img, template_imgs, equipment_name):
    """Match a single diagram segment with equipment templates using multiple methods"""
    best_match = None
    best_confidence = 0

    for i, template_img in enumerate(template_imgs):
        try:
            # Method 1: Multi-scale template matching
            template_result = multiscale_template_matching_segment(segment_img, template_img)
            if template_result and template_result['confidence'] > best_confidence:
                best_confidence = template_result['confidence']
                best_match = {
                    **template_result,
                    'method': 'TEMPLATE_SEGMENT',
                    'template_index': i
                }

            # Method 2: Feature-based matching
            feature_result = feature_based_matching_segment(segment_img, template_img)
            if feature_result and feature_result['confidence'] > best_confidence:
                best_confidence = feature_result['confidence']
                best_match = {
                    **feature_result,
                    'method': 'FEATURE_SEGMENT',
                    'template_index': i
                }

            # Method 3: Structural similarity
            ssim_result = structural_similarity_matching(segment_img, template_img)
            if ssim_result and ssim_result['confidence'] > best_confidence:
                best_confidence = ssim_result['confidence']
                best_match = {
                    **ssim_result,
                    'method': 'SSIM_SEGMENT',
                    'template_index': i
                }

        except Exception as e:
            print(f"Error matching segment with template {i}: {e}")
            continue

    return best_match

def multiscale_template_matching_segment(segment, template):
    """Improved template matching for segments"""
    try:
        best_match = None
        best_confidence = 0

        scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]

        for scale in scales:
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)

            if scaled_w >= segment.shape[1] or scaled_h >= segment.shape[0]:
                continue
            if scaled_w < 20 or scaled_h < 20:
                continue

            scaled_template = cv2.resize(template, (scaled_w, scaled_h))

            # Template matching
            result = cv2.matchTemplate(segment, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.4 and max_val > best_confidence:
                best_confidence = max_val
                best_match = {
                    'confidence': max_val,
                    'location': max_loc,
                    'scale': scale,
                    'width': scaled_w,
                    'height': scaled_h
                }

        return best_match
    except Exception as e:
        print(f"Error in segment template matching: {e}")
        return None

def feature_based_matching_segment(segment, template):
    """SIFT/ORB matching for segments"""
    try:
        # Try SIFT first
        sift = cv2.SIFT_create(nfeatures=500)
        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(segment, None)

        if des1 is not None and des2 is not None and len(des1) > 4 and len(des2) > 4:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 4:
                # Calculate confidence based on matches
                confidence = min(len(good_matches) / max(len(des1), len(des2)), 1.0)

                if confidence > 0.1:
                    return {
                        'confidence': confidence,
                        'location': (0, 0),  # Full segment match
                        'width': segment.shape[1],
                        'height': segment.shape[0],
                        'matches_count': len(good_matches)
                    }
    except:
        pass

    return None

def structural_similarity_matching(segment, template):
    """Structural similarity matching"""
    try:
        # Resize template to match segment size
        template_resized = cv2.resize(template, (segment.shape[1], segment.shape[0]))

        # Calculate mean squared error
        mse = np.mean((segment.astype(float) - template_resized.astype(float)) ** 2)

        # Convert to similarity score
        max_possible_mse = 255 * 255
        similarity = 1 - (mse / max_possible_mse)

        if similarity > 0.3:
            return {
                'confidence': similarity,
                'location': (0, 0),
                'width': segment.shape[1],
                'height': segment.shape[0]
            }
    except:
        pass

    return None

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
        SELECT e.id, e.name, GROUP_CONCAT(ei.image_path) as images
        FROM equipment e
        LEFT JOIN equipment_images ei ON e.id = ei.equipment_id
        GROUP BY e.id, e.name
    ''')

    equipment = []
    for row in cursor.fetchall():
        images = row[2].split(',') if row[2] else []
        equipment.append({
            'id': row[0],
            'name': row[1],
            'images': images
        })

    conn.close()
    return jsonify(equipment)

@app.route('/api/equipment/<int:equipment_id>', methods=['DELETE'])
def delete_equipment(equipment_id):
    """Delete equipment by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get equipment images before deleting
    cursor.execute('SELECT image_path FROM equipment_images WHERE equipment_id = ?', (equipment_id,))
    image_paths = cursor.fetchall()

    # Delete equipment images from database
    cursor.execute('DELETE FROM equipment_images WHERE equipment_id = ?', (equipment_id,))

    # Delete equipment from database
    cursor.execute('DELETE FROM equipment WHERE id = ?', (equipment_id,))

    conn.commit()
    conn.close()

    # Delete actual image files
    for (image_path,) in image_paths:
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file {image_path}: {e}")

    return jsonify({'message': 'Equipment deleted successfully'})

@app.route('/api/equipment', methods=['POST'])
def register_equipment():
    """Register new equipment with images"""
    if 'name' not in request.form:
        return jsonify({'error': 'Equipment name is required'}), 400

    name = request.form['name']

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Insert equipment
    cursor.execute('INSERT INTO equipment (name) VALUES (?)', (name,))
    equipment_id = cursor.lastrowid

    # Handle uploaded images
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

    # Convert PDF to image if it's a PDF
    image_path = None
    if filename.lower().endswith('.pdf'):
        try:
            images = convert_from_path(filepath)
            if images:
                image_filename = f"{os.path.splitext(filename)[0]}.png"
                full_image_path = os.path.join(STATIC_FOLDER, 'diagrams', image_filename)
                images[0].save(full_image_path)
                image_path = f"static/diagrams/{image_filename}"  # Relative path for URL
        except Exception as e:
            return jsonify({'error': f'Failed to convert PDF: {str(e)}'}), 500
    else:
        # Copy image to static folder
        image_filename = filename
        full_image_path = os.path.join(STATIC_FOLDER, 'diagrams', image_filename)
        import shutil
        shutil.copy2(filepath, full_image_path)
        image_path = f"static/diagrams/{image_filename}"  # Relative path for URL

    # Save to database
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
    cursor.execute('SELECT id, name, file_path, image_path FROM diagrams')

    diagrams = []
    for row in cursor.fetchall():
        diagrams.append({
            'id': row[0],
            'name': row[1],
            'file_path': row[2],
            'image_path': row[3]
        })

    conn.close()
    return jsonify(diagrams)

@app.route('/api/diagrams/<int:diagram_id>', methods=['DELETE'])
def delete_diagram(diagram_id):
    """Delete a diagram by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Get diagram info before deleting
    cursor.execute('SELECT file_path, image_path FROM diagrams WHERE id = ?', (diagram_id,))
    result = cursor.fetchone()

    if not result:
        conn.close()
        return jsonify({'error': 'Diagram not found'}), 404

    file_path, image_path = result

    # Delete from database
    cursor.execute('DELETE FROM diagrams WHERE id = ?', (diagram_id,))
    conn.commit()
    conn.close()

    # Delete actual files
    try:
        # Delete original file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        # Delete image file (handle both full path and relative path)
        if image_path:
            if image_path.startswith('static/'):
                # Relative path
                full_image_path = os.path.join(BASE_DIR, image_path)
            else:
                # Full path (legacy)
                full_image_path = image_path

            if os.path.exists(full_image_path):
                os.remove(full_image_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

    return jsonify({'message': 'Diagram deleted successfully'})

@app.route('/api/match-equipment', methods=['POST'])
def match_equipment():
    """Advanced segment-based equipment matching"""
    data = request.get_json()

    if not data or 'diagram_path' not in data or 'equipment_ids' not in data:
        return jsonify({'error': 'Missing diagram_path or equipment_ids'}), 400

    diagram_path = data['diagram_path']
    equipment_ids = data['equipment_ids']

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

        print(f"=== SEGMENT-BASED MATCHING ===")
        print(f"Diagram size: {diagram_img.shape[1]}x{diagram_img.shape[0]}")

        # Step 1: Segment diagram into regions
        segments = segment_diagram_regions(diagram_img)
        print(f"Found {len(segments)} segments in diagram")

        matches = []

        # Get equipment data from database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        for equipment_id in equipment_ids:
            # Get equipment images
            cursor.execute('''
                SELECT e.name, ei.image_path
                FROM equipment e
                JOIN equipment_images ei ON e.id = ei.equipment_id
                WHERE e.id = ?
            ''', (equipment_id,))

            equipment_results = cursor.fetchall()

            if not equipment_results:
                print(f"No images found for equipment ID {equipment_id}")
                continue

            equipment_name = equipment_results[0][0]
            print(f"\n--- Processing equipment: {equipment_name} ---")

            # Load all template images for this equipment
            template_imgs = []
            for _, template_path in equipment_results:
                if os.path.exists(template_path):
                    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template_img is not None:
                        template_imgs.append(template_img)
                        print(f"  Loaded template: {os.path.basename(template_path)} ({template_img.shape[1]}x{template_img.shape[0]})")

            if not template_imgs:
                print(f"  No valid templates found for {equipment_name}")
                continue

            # Step 2: Match each segment against equipment templates
            best_overall_match = None
            best_overall_confidence = 0
            best_segment_index = -1

            for seg_idx, segment_data in enumerate(segments):
                segment_img = segment_data['region']
                segment_bbox = segment_data['bbox']

                print(f"  Testing segment {seg_idx}: {segment_bbox} (area: {segment_data['area']})")

                # Match this segment with equipment templates
                match_result = match_segment_with_equipment(segment_img, template_imgs, equipment_name)

                if match_result and match_result['confidence'] > best_overall_confidence:
                    best_overall_confidence = match_result['confidence']
                    best_overall_match = match_result
                    best_segment_index = seg_idx

                    print(f"    ✓ Good match! Confidence: {match_result['confidence']:.3f} Method: {match_result['method']}")
                else:
                    print(f"    ✗ No match (confidence: {match_result['confidence'] if match_result else 'N/A'})")

            # Step 3: Create final match if found
            if best_overall_match and best_overall_confidence > 0.4:  # Higher threshold for segment matching
                segment_bbox = segments[best_segment_index]['bbox']

                # Calculate absolute coordinates
                if best_overall_match['location'] == (0, 0):
                    # Full segment match
                    abs_x, abs_y = segment_bbox[0], segment_bbox[1]
                    abs_w, abs_h = segment_bbox[2], segment_bbox[3]
                else:
                    # Partial segment match
                    abs_x = segment_bbox[0] + best_overall_match['location'][0]
                    abs_y = segment_bbox[1] + best_overall_match['location'][1]
                    abs_w = best_overall_match['width']
                    abs_h = best_overall_match['height']

                final_match = {
                    'equipment_id': equipment_id,
                    'equipment_name': equipment_name,
                    'x': int(abs_x),
                    'y': int(abs_y),
                    'width': int(abs_w),
                    'height': int(abs_h),
                    'confidence': float(best_overall_confidence),
                    'center_x': int(abs_x + abs_w // 2),
                    'center_y': int(abs_y + abs_h // 2),
                    'method': best_overall_match['method'],
                    'segment_index': best_segment_index
                }

                matches.append(final_match)
                print(f"  ✅ FINAL MATCH: {equipment_name} at segment {best_segment_index}")
                print(f"     Location: ({abs_x}, {abs_y}) Size: {abs_w}x{abs_h}")
                print(f"     Confidence: {best_overall_confidence:.3f} Method: {best_overall_match['method']}")
            else:
                print(f"  ❌ No adequate match found for {equipment_name} (best confidence: {best_overall_confidence:.3f})")

        conn.close()

        print(f"\n=== MATCHING COMPLETE ===")
        print(f"Total matches found: {len(matches)}")

        return jsonify({
            'matches': matches,
            'total_found': len(matches),
            'segments_processed': len(segments)
        })

    except Exception as e:
        print(f"Error in segment-based matching: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segment-based matching failed: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_FOLDER, 'diagrams'), exist_ok=True)
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

    app.run(debug=False, host='0.0.0.0', port=8000)