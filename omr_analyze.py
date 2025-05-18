import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.perspective import four_point_transform
import os
import json

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
os.makedirs('output/detect_omr', exist_ok=True)
# Remove all files from output/detect_omr directory
if os.path.exists('output/detect_omr'):
    for file in os.listdir('output/detect_omr'):
        file_path = os.path.join('output/detect_omr', file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Read image
image = cv2.imread("assets/scanned_omr_9.png")
if image is None:
    print("Error: Could not read the image file")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect markers with improved parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshConstant = 7
detector = aruco.ArucoDetector(aruco_dict, params)
corners, ids, rejected = detector.detectMarkers(gray)
print(ids)
print(corners)
if ids is not None:
    print(f"Found {len(ids)} markers with IDs: {[id[0] for id in ids]}")
    
    if len(ids) == 4:
        # Create a mapping of ID to corners
        id_corner_map = {}
        for marker_id, marker_corners in zip(ids, corners):
            # Store all corners for each marker
            # ArUco corners are in clockwise order: top-left, top-right, bottom-right, bottom-left
            id_corner_map[marker_id[0]] = marker_corners[0]
        
        # Sort corners in TL, TR, BR, BL order using IDs
        ordered_ids = [1, 2, 4, 3]  # [TL, TR, BR, BL]
        try:
            # Get the outer corners of each marker based on their position
            ordered_pts = np.array([
                id_corner_map[1][0],  # Top-left marker's top-left corner
                id_corner_map[2][1],  # Top-right marker's top-right corner
                id_corner_map[4][2],  # Bottom-right marker's bottom-right corner
                id_corner_map[3][3]   # Bottom-left marker's bottom-left corner
            ], dtype=np.float32)
            
            # Draw detected markers for debugging
            debug_img = image.copy()
            for marker_id, marker_corners in zip(ids, corners):
                # Draw marker boundaries
                cv2.polylines(debug_img, [marker_corners[0].astype(np.int32)], True, (0, 255, 0), 2)
                # Add marker ID
                center = marker_corners[0].mean(axis=0).astype(np.int32)
                cv2.putText(debug_img, f"ID: {marker_id[0]}", tuple(center), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw and label each corner of the marker
                for idx, corner in enumerate(marker_corners[0]):
                    corner_pt = tuple(corner.astype(int))
                    cv2.circle(debug_img, corner_pt, 4, (255, 0, 0), -1)
                    cv2.putText(debug_img, f"{marker_id[0]}:{idx}", 
                              (corner_pt[0] + 5, corner_pt[1] + 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw the warping points
            for idx, pt in enumerate(ordered_pts):
                pt_int = tuple(pt.astype(int))
                cv2.circle(debug_img, pt_int, 8, (0, 255, 255), -1)
                cv2.putText(debug_img, f"Warp {idx}", 
                          (pt_int[0] + 5, pt_int[1] + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imwrite("output/detect_omr/detected_markers.jpg", debug_img)
            print("Saved debug image with detected markers")
            
            # Calculate padding (1% of image dimensions)
            h, w = image.shape[:2]
            pad_x = int(w * 0.01)
            pad_y = int(h * 0.01)
            
            # Remove padding: use marker corners directly for warping
            # ordered_pts[0] = ordered_pts[0] - [pad_x, pad_y]  # Top-left
            # ordered_pts[1] = ordered_pts[1] + [pad_x, -pad_y]  # Top-right
            # ordered_pts[2] = ordered_pts[2] + [pad_x, pad_y]   # Bottom-right
            # ordered_pts[3] = ordered_pts[3] + [-pad_x, pad_y]  # Bottom-left
            # Warp the image
            warped = four_point_transform(image, ordered_pts)
            
            # Save both original size and a fixed-size version
            cv2.imwrite("output/detect_omr/warped_omr.jpg", warped)

            # Always resize to fixed size (1030x1500)
            fixed_size = (1030, 1500)  # width x height
            warped_resized = cv2.resize(warped, fixed_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite("output/detect_omr/warped_omr_resized.jpg", warped_resized)
            print("Saved both original and fixed-size warped images!")
            
        except KeyError as e:
            print(f"Error: Missing marker with ID {e}. Make sure all required markers (1,2,3,4) are visible.")
        except Exception as e:
            print(f"Error during warping: {e}")
    else:
        print(f"Error: Need exactly 4 markers, but found {len(ids)}")
        # Save debug image even when not all markers are found
        debug_img = image.copy()
        if ids is not None:
            for marker_id, marker_corners in zip(ids, corners):
                cv2.polylines(debug_img, [marker_corners[0].astype(np.int32)], True, (0, 255, 0), 2)
                center = marker_corners[0].mean(axis=0).astype(np.int32)
                cv2.putText(debug_img, f"ID: {marker_id[0]}", tuple(center), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite("output/detect_omr/partial_detection.jpg", debug_img)
        print("Saved debug image showing partial marker detection")
else:
    print("No markers detected!")



os.makedirs('output', exist_ok=True)
os.makedirs('output/omr_grader', exist_ok=True)
# Remove all files from output/omr_grader directory
if os.path.exists('output/omr_grader'):
    for file in os.listdir('output/omr_grader'):
        file_path = os.path.join('output/omr_grader', file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# === Configuration ===
IMAGE_PATH = "output/detect_omr/warped_omr_resized.jpg"
JSON_PATH = "assets/omr_coordinates.json"
FILL_THRESHOLD = 0.85  # Very high threshold for complete circle filling
CONFIDENCE_THRESHOLD = 0.9  # High confidence required
MIN_FILLED_AREA = 0.8  # Minimum area that must be filled
DEBUG_MODE = True  # Set to True to save debug images
DIGIT_HEIGHT_GAP = 20  # Pixel gap between digits stacked vertically

# Create debug directory if needed
if DEBUG_MODE:
    os.makedirs('output/omr_grader', exist_ok=True)

# === Enhanced Image Preprocessing ===
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
    
    # Apply strong Gaussian blur to remove text
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply binary threshold with high value to detect only dark fills
    _, thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Remove noise and small elements
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    # Dilate to ensure filled areas are well connected
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    if DEBUG_MODE:
        # Save intermediate steps for debugging
        cv2.imwrite('output/omr_grader/1_gray.jpg', gray)
        cv2.imwrite('output/omr_grader/2_blur.jpg', blur)
        cv2.imwrite('output/omr_grader/3_thresh.jpg', thresh1)
        cv2.imwrite('output/omr_grader/4_preprocessed.jpg', thresh)
    
    return thresh

# === Enhanced Bubble Detection ===
def is_filled(img, x, y, r, threshold=FILL_THRESHOLD):
    # Get the full circle area
    roi = img[y - r:y + r, x - r:x + r]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return False, 0.0
    
    # Create circular mask for the entire circle
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    
    # Calculate filled ratio
    total_pixels = np.sum(mask == 255)
    filled_pixels = np.sum((roi == 255) & (mask == 255))
    fill_ratio = filled_pixels / total_pixels
    
    # Calculate confidence score
    confidence = min(1.0, fill_ratio / threshold)
    
    # For complete circle filling, we need a very high fill ratio
    is_completely_filled = fill_ratio > threshold
    
    return is_completely_filled, confidence

# === Load and Process Image ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Could not read image at {IMAGE_PATH}")

thresh = preprocess_image(img)

# === Load bubble coordinates ===
with open(JSON_PATH, "r") as f:
    data = json.load(f)

results = {
    "student_id": "",
    "paper_code": "",
    "answers": {},
    "confidence_scores": {}
}

# === Detect student ID digits ===
for letter_key, bubbles in data.get("student_id", {}).items():
    max_confidence = 0
    selected_digit = "X"
    
    # Each letter position has 7 bubbles (0-6)
    for i, bubble in enumerate(bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            # The bubble index (0-6) directly corresponds to the digit value
            selected_digit = str(i)  # i will be 0-6 representing the actual digit
    
    # Only add the digit if we have a valid detection
    if max_confidence >= CONFIDENCE_THRESHOLD:
        results["student_id"] += selected_digit
    else:
        results["student_id"] += "X"  # Mark as undetected if confidence is too low
    
    results["confidence_scores"][f"student_id_{letter_key}"] = max_confidence

# === Detect Paper Code digits ===
for letter_key, bubbles in data.get("paper_code", {}).items():
    max_confidence = 0
    selected_digit = "X"
    
    # Each letter position has 7 bubbles (0-6)
    for i, bubble in enumerate(bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            # The bubble index (0-6) directly corresponds to the digit value
            selected_digit = str(i)  # i will be 0-6 representing the actual digit
    
    # Only add the digit if we have a valid detection
    if max_confidence >= CONFIDENCE_THRESHOLD:
        results["paper_code"] += selected_digit
    else:
        results["paper_code"] += "X"  # Mark as undetected if confidence is too low
    
    results["confidence_scores"][f"paper_code_{letter_key}"] = max_confidence

# === Detect MCQ answers ===
for q_key, q_bubbles in data.get("questions", {}).items():
    marked_answers = []
    confidences = []
    for i, bubble in enumerate(q_bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
            marked_answers.append(chr(65 + i))  # A, B, C, D, ...
            confidences.append(confidence)
    if not marked_answers:
        results["answers"][q_key] = []
        results["confidence_scores"][q_key] = 0.0
    else:
        results["answers"][q_key] = marked_answers
        results["confidence_scores"][q_key] = max(confidences)

# === Output result ===
print(json.dumps(results, indent=2))

# === Save debug visualization if enabled ===
if DEBUG_MODE:
    debug_img = img.copy()
    
    # Draw student ID bubbles
    for letter_key, bubbles in data.get("student_id", {}).items():
        for i, bubble in enumerate(bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            # Draw the circle
            cv2.circle(debug_img, (x, y), r, (255, 0, 0), 2)  # Blue for student ID
            
            # Get the fill status for visualization
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            
            # Color based on detection and confidence
            if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green for detected
                cv2.circle(debug_img, (x, y), r-2, color, -1)  # Fill the detected circle
            else:
                color = (0, 0, 255)  # Red for not detected
            
            # Add digit label and confidence
            cv2.putText(debug_img, f"{i}", (x - r, y - r - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw paper code bubbles
    for letter_key, bubbles in data.get("paper_code", {}).items():
        for i, bubble in enumerate(bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            # Draw the circle
            cv2.circle(debug_img, (x, y), r, (255, 255, 0), 2)  # Yellow for paper code
            
            # Get the fill status for visualization
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            
            # Color based on detection and confidence
            if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green for detected
                cv2.circle(debug_img, (x, y), r-2, color, -1)  # Fill the detected circle
            else:
                color = (0, 0, 255)  # Red for not detected
            
            # Add digit label and confidence
            cv2.putText(debug_img, f"{i}", (x - r, y - r - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw question bubbles
    for q_key, q_bubbles in data.get("questions", {}).items():
        for i, bubble in enumerate(q_bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            # Draw the circle
            cv2.circle(debug_img, (x, y), r, (0, 0, 255), 2)  # Red for questions
            
            # Get the fill status for visualization
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            
            # Color based on detection and confidence
            if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)  # Green for detected
                cv2.circle(debug_img, (x, y), r-2, color, -1)  # Fill the detected circle
            else:
                color = (0, 0, 255)  # Red for not detected
            
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite('output/omr_grader/detected_bubbles.jpg', debug_img)

