from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.perspective import four_point_transform
import json
import os
from io import BytesIO
from typing import Optional
from pydantic import BaseModel
import base64

app = FastAPI()

# Load static JSON once on startup
JSON_PATH = "assets/omr_coordinates.json"
with open(JSON_PATH, "r") as f:
    DEFAULT_OMR_COORDINATES = json.load(f)

# Your constants, thresholds, debug flags
FILL_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD = 0.9
DEBUG_MODE = False  # You can enable for debugging

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def is_filled(img, x, y, r, threshold=FILL_THRESHOLD):
    roi = img[y - r:y + r, x - r:x + r]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return False, 0.0
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    total_pixels = np.sum(mask == 255)
    filled_pixels = np.sum((roi == 255) & (mask == 255))
    fill_ratio = filled_pixels / total_pixels
    confidence = min(1.0, fill_ratio / threshold)
    is_completely_filled = fill_ratio > threshold
    return is_completely_filled, confidence

def detect_aruco_and_warp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshConstant = 7
    detector = aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) == 4:
        id_corner_map = {marker_id[0]: marker_corners[0] for marker_id, marker_corners in zip(ids, corners)}
        ordered_ids = [1, 2, 4, 3]
        ordered_pts = np.array([
            id_corner_map[1][0],
            id_corner_map[2][1],
            id_corner_map[4][2],
            id_corner_map[3][3]
        ], dtype=np.float32)
        warped = four_point_transform(image, ordered_pts)
        warped_resized = cv2.resize(warped, (1030, 1500), interpolation=cv2.INTER_AREA)
        return warped_resized
    else:
        raise ValueError("Exactly 4 ArUco markers with IDs 1,2,3,4 are required.")

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def generate_debug_image(img, thresh, omr_coordinates, confidence_threshold):
    debug_img = img.copy()
    # Draw student ID bubbles
    for letter_key, bubbles in omr_coordinates.get("student_id", {}).items():
        for i, bubble in enumerate(bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            color = (0, 255, 0) if is_filled_bubble and confidence >= confidence_threshold else (0, 0, 255)
            cv2.circle(debug_img, (x, y), r, color, 2)
            cv2.putText(debug_img, f"{i}", (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Draw paper code bubbles
    for letter_key, bubbles in omr_coordinates.get("paper_code", {}).items():
        for i, bubble in enumerate(bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            color = (0, 255, 0) if is_filled_bubble and confidence >= confidence_threshold else (0, 0, 255)
            cv2.circle(debug_img, (x, y), r, color, 2)
            cv2.putText(debug_img, f"{i}", (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Draw question bubbles
    for q_key, q_bubbles in omr_coordinates.get("questions", {}).items():
        for i, bubble in enumerate(q_bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            color = (0, 255, 0) if is_filled_bubble and confidence >= confidence_threshold else (0, 0, 255)
            cv2.circle(debug_img, (x, y), r, color, 2)
            cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return debug_img

@app.post("/process-omr/")
async def process_omr(
    omr_image: UploadFile = File(...),
    coordinates: Optional[str] = Form(None)
):
    try:
        # Read image bytes from upload
        contents = await omr_image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Use provided coordinates if available, otherwise use default
        omr_coordinates = DEFAULT_OMR_COORDINATES
        if coordinates:
            try:
                omr_coordinates = json.loads(coordinates)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON coordinates format")

        # Detect and warp
        warped_img = detect_aruco_and_warp(image)

        # Preprocess
        thresh = preprocess_image(warped_img)

        # Process OMR bubbles as per your logic
        results = {
            "student_id": "",
            "paper_code": "",
            "answers": {},
            "confidence_scores": {}
        }

        # Student ID
        for letter_key, bubbles in omr_coordinates.get("student_id", {}).items():
            max_confidence = 0
            selected_digit = "X"
            for i, bubble in enumerate(bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                if is_filled_bubble and confidence > max_confidence:
                    max_confidence = confidence
                    selected_digit = str(i)
            if max_confidence >= CONFIDENCE_THRESHOLD:
                results["student_id"] += selected_digit
            else:
                results["student_id"] += "X"
            results["confidence_scores"][f"student_id_{letter_key}"] = max_confidence

        # Paper Code
        for letter_key, bubbles in omr_coordinates.get("paper_code", {}).items():
            max_confidence = 0
            selected_digit = "X"
            for i, bubble in enumerate(bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                if is_filled_bubble and confidence > max_confidence:
                    max_confidence = confidence
                    selected_digit = str(i)
            if max_confidence >= CONFIDENCE_THRESHOLD:
                results["paper_code"] += selected_digit
            else:
                results["paper_code"] += "X"
            results["confidence_scores"][f"paper_code_{letter_key}"] = max_confidence

        # Questions
        for i in range(1, 71):
            q_key = f"question_{i}"
            q_bubbles = omr_coordinates.get("questions", {}).get(q_key, [])
            marked_answers = []
            confidences = []
            for j, bubble in enumerate(q_bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                    marked_answers.append(chr(65 + j))
                    confidences.append(confidence)
            if not marked_answers:
                results["answers"][q_key] = []
                results["confidence_scores"][q_key] = 0.0
            else:
                results["answers"][q_key] = marked_answers
                results["confidence_scores"][q_key] = max(confidences)

        print("Question keys:", list(omr_coordinates.get("questions", {}).keys()))

        results["totalQuestions"] = len(results["answers"])
        results["answeredQuestions"] = len([q for q in results["answers"] if results["answers"][q]])
        # You can add correct/incorrect logic here as needed

        debug_img = generate_debug_image(warped_img, thresh, omr_coordinates, CONFIDENCE_THRESHOLD)
        _, buffer = cv2.imencode('.jpg', debug_img)
        results["debug_image_base64"] = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))