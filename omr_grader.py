import cv2
import numpy as np
import json
import os

# === Configuration ===
IMAGE_PATH = "output/detect_omr/warped_omr.jpg"
JSON_PATH = "assets/omr_coordinates.json"
FILL_THRESHOLD = 0.45        # Significantly lowered from 0.65 to be more sensitive
CONFIDENCE_THRESHOLD = 0.4    # Lowered from 0.7 to catch more filled bubbles
MIN_FILLED_AREA = 0.7  # Adjust if needed
DEBUG_MODE = True  # Set to True to save debug images
DIGIT_HEIGHT_GAP = 10  # Pixel gap between digits stacked vertically

# Adjust these values for better detection
THRESHOLD_VALUE = 160        # Lowered from 170 to catch lighter marks
CONTRAST_ALPHA = 1.5         # Increased contrast
CONTRAST_BETA = 20          # Adjusted for better balance

# Create debug directory if needed
if DEBUG_MODE:
    os.makedirs('output/omr_grader', exist_ok=True)

# === Enhanced Image Preprocessing ===
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast more aggressively
    gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)
    
    # Apply adaptive thresholding with more sensitive parameters
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 19, 8)  # Adjusted block size and C
    
    # Remove noise while preserving filled bubbles
    kernel = np.ones((2,2), np.uint8)  # Smaller kernel
    thresh = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    # Light dilation to connect nearby pixels
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    if DEBUG_MODE:
        cv2.imwrite('output/omr_grader/1_gray.jpg', gray)
        cv2.imwrite('output/omr_grader/2_thresh.jpg', thresh1)
        cv2.imwrite('output/omr_grader/3_preprocessed.jpg', thresh)
    
    return thresh

# === Enhanced Bubble Detection ===
def is_filled(img, x, y, r, threshold=FILL_THRESHOLD):
    # Increase padding to capture more of the bubble area
    padding = 3  # Increased from 2
    roi = img[y - r - padding:y + r + padding, x - r - padding:x + r + padding]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return False, 0.0
    
    # Create circular mask with slightly smaller radius to focus on center
    mask = np.zeros(roi.shape, dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    cv2.circle(mask, center, r-1, 255, -1)  # Slightly smaller radius
    
    # Calculate filled ratio
    total_pixels = np.sum(mask == 255)
    filled_pixels = np.sum((roi == 255) & (mask == 255))
    fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
    
    # More sophisticated confidence calculation
    base_confidence = fill_ratio / threshold
    # Boost confidence if fill_ratio is close to threshold
    if 0.8 * threshold <= fill_ratio <= 1.2 * threshold:
        base_confidence *= 1.2
    
    confidence = min(1.0, base_confidence)
    is_completely_filled = fill_ratio > threshold
    
    if DEBUG_MODE:
        # Save ROI for all bubbles to analyze detection
        debug_roi = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_roi, center, r-1, (0, 255, 0), 1)
        cv2.putText(debug_roi, f"{fill_ratio:.2f}", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        os.makedirs('output/omr_grader/bubble_debug', exist_ok=True)
        cv2.imwrite(f'output/omr_grader/bubble_debug/bubble_{x}_{y}.jpg', debug_roi)
    
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
if "student_id" in data:
    student_id = ""
    max_confidence = 0
    
    # Iterate through each digit position (7 digits)
    for i in range(1, 8):
        letter_key = f"letter_{i}"
        if letter_key in data["student_id"]:
            bubbles = data["student_id"][letter_key]
            max_confidence_digit = 0
            selected_digit = "X"
            
            # Sort bubbles by y-coordinate (top to bottom)
            sorted_bubbles = sorted(bubbles, key=lambda b: b["y"])
            
            # Check each bubble in the column (0-9)
            for idx, bubble in enumerate(sorted_bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                
                # More lenient detection logic
                if is_filled_bubble and confidence > max_confidence_digit:
                    max_confidence_digit = confidence
                    selected_digit = str(idx)
                elif confidence > 0.35 and max_confidence_digit == 0:  # Even more lenient backup check
                    max_confidence_digit = confidence
                    selected_digit = str(idx)
            
            # More lenient threshold for accepting a digit
            student_id += selected_digit if max_confidence_digit > 0.35 else "X"
            max_confidence = max(max_confidence, max_confidence_digit)
    
    results["student_id"] = student_id
    results["confidence_scores"]["student_id"] = max_confidence

    # Enhanced debug visualization with more details
    if DEBUG_MODE:
        student_id_debug = img.copy()
        for letter_key, bubbles in data["student_id"].items():
            sorted_bubbles = sorted(bubbles, key=lambda b: b["y"])
            for idx, bubble in enumerate(sorted_bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                
                # More detailed color coding
                if is_filled_bubble:
                    color = (0, 255, 0)  # Green for clear detection
                elif confidence > 0.35:
                    color = (0, 255, 255)  # Yellow for partial detection
                elif confidence > 0.25:
                    color = (255, 165, 0)  # Orange for weak detection
                else:
                    color = (0, 0, 255)  # Red for no detection
                
                cv2.circle(student_id_debug, (x, y), r, color, 2)
                # Add more detailed information
                cv2.putText(student_id_debug, f"{idx}:{confidence:.3f}", 
                          (x - r, y - r - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add position labels and detected values
        for i in range(1, 8):
            letter_key = f"letter_{i}"
            if letter_key in data["student_id"]:
                bubbles = data["student_id"][letter_key]
                first_bubble = bubbles[0]
                detected_value = student_id[i-1]
                cv2.putText(student_id_debug, f"D{i}={detected_value}", 
                          (first_bubble["x"] - 25, first_bubble["y"] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imwrite('output/omr_grader/student_id_debug.jpg', student_id_debug)

# === Detect paper code ===
if "paper_code" in data:
    paper_code = ""
    max_confidence = 0
    
    # Iterate through each position
    for i in range(1, 8):  # 7 digits for paper code
        letter_key = f"letter_{i}"
        if letter_key in data["paper_code"]:
            bubbles = data["paper_code"][letter_key]
            max_confidence_digit = 0
            selected_digit = "X"
            
            # Sort bubbles by y-coordinate (top to bottom)
            sorted_bubbles = sorted(bubbles, key=lambda b: b["y"])
            
            # Check each bubble in the column (0-9)
            for idx, bubble in enumerate(sorted_bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                
                if is_filled_bubble and confidence > max_confidence_digit:
                    max_confidence_digit = confidence
                    selected_digit = str(idx)  # Use vertical position (0-9) as the digit
            
            paper_code += selected_digit if max_confidence_digit > 0.4 else "X"
            max_confidence = max(max_confidence, max_confidence_digit)
    
    results["paper_code"] = paper_code
    results["confidence_scores"]["paper_code"] = max_confidence

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
