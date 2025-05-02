import cv2
import numpy as np
import json
import os

# === Configuration ===
IMAGE_PATH = "output/detect_omr/warped_omr.jpg"
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
    "exam_date": "",
    "answers": {},
    "confidence_scores": {}
}

# === Detect student ID digits ===
for bubble in data.get("student_id", []):
    max_confidence = 0
    selected_digit = "X"
    
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            selected_digit = str(digit)
    
    results["student_id"] += selected_digit
    results["confidence_scores"]["student_id"] = max_confidence

# === Detect exam date digits ===
for bubble in data.get("exam_date", []):
    max_confidence = 0
    selected_digit = "X"
    
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            selected_digit = str(digit)
    
    results["exam_date"] += selected_digit
    results["confidence_scores"]["exam_date"] = max_confidence

# === Detect MCQ answers ===
for q_key, q_bubbles in data.get("questions", {}).items():
    max_confidence = 0
    selected_answer = "Not marked"
    multiple_marks = False
    
    for i, bubble in enumerate(q_bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble:
            if confidence > max_confidence:
                max_confidence = confidence
                selected_answer = chr(65 + i)  # A, B, C, D
            else:
                multiple_marks = True
    
    if max_confidence < CONFIDENCE_THRESHOLD:
        selected_answer = "Not marked"
    elif multiple_marks:
        selected_answer = "Multiple marks detected"
    
    results["answers"][q_key] = selected_answer
    results["confidence_scores"][q_key] = max_confidence

# === Output result ===
print(json.dumps(results, indent=2))

# === Save debug visualization if enabled ===
if DEBUG_MODE:
    debug_img = img.copy()
    for q_key, q_bubbles in data.get("questions", {}).items():
        for i, bubble in enumerate(q_bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            # Draw the circle
            cv2.circle(debug_img, (x, y), r, (0, 0, 255), 2)
            
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
