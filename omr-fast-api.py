from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import json
from typing import Dict, Any
import shutil
from pathlib import Path
import requests
from pydantic import BaseModel

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Initializing FastAPI app")
# Initialize FastAPI app
app = FastAPI(
    title="OMR Grading System API",
    description="API for automatic OMR sheet detection and grading",
    version="1.0.0"
)
logger.debug("FastAPI app initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
FILL_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD = 0.9
MIN_FILLED_AREA = 0.8
DEBUG_MODE = True
DIGIT_HEIGHT_GAP = 20

# Create necessary directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/omr_grader', exist_ok=True)
os.makedirs('output/detect_omr', exist_ok=True)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh1 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    if DEBUG_MODE:
        cv2.imwrite('output/omr_grader/1_gray.jpg', gray)
        cv2.imwrite('output/omr_grader/2_blur.jpg', blur)
        cv2.imwrite('output/omr_grader/3_thresh.jpg', thresh1)
        cv2.imwrite('output/omr_grader/4_preprocessed.jpg', thresh)
    
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

def process_omr(image, coordinates_data):
    thresh = preprocess_image(image)
    
    results = {
        "student_id": "",
        "paper_code": "",
        "answers": {},
        "confidence_scores": {}
    }
    
    # Process student ID
    for letter_key, bubbles in coordinates_data.get("student_id", {}).items():
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
    
    # Process paper code
    for letter_key, bubbles in coordinates_data.get("paper_code", {}).items():
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
    
    # Process questions
    for q_key, q_bubbles in coordinates_data.get("questions", {}).items():
        marked_answers = []
        confidences = []
        for i, bubble in enumerate(q_bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            is_filled_bubble, confidence = is_filled(thresh, x, y, r)
            if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                marked_answers.append(chr(65 + i))
                confidences.append(confidence)
        
        results["answers"][q_key] = marked_answers
        results["confidence_scores"][q_key] = max(confidences) if confidences else 0.0
    
    if DEBUG_MODE:
        debug_img = image.copy()
        
        # Draw debug visualization
        for letter_key, bubbles in coordinates_data.get("student_id", {}).items():
            for i, bubble in enumerate(bubbles):
                x, y, r = bubble["x"], bubble["y"], bubble["r"]
                cv2.circle(debug_img, (x, y), r, (255, 0, 0), 2)
                
                is_filled_bubble, confidence = is_filled(thresh, x, y, r)
                color = (0, 255, 0) if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
                
                if is_filled_bubble and confidence >= CONFIDENCE_THRESHOLD:
                    cv2.circle(debug_img, (x, y), r-2, color, -1)
                
                cv2.putText(debug_img, f"{i}", (x - r, y - r - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(debug_img, f"{confidence:.2f}", (x - r, y - r - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Similar debug visualization for paper code and questions...
        cv2.imwrite('output/omr_grader/detected_bubbles.jpg', debug_img)
    
    return results

class S3OMRRequest(BaseModel):
    omr_image_url: str
    coordinates_file_url: str

@app.post("/grade-omr-from-s3")
async def grade_omr_from_s3(request: S3OMRRequest):
    try:
        # Download image from S3 URL
        image_response = requests.get(request.omr_image_url, stream=True)
        if image_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download OMR image from S3")
        
        # Convert image data to OpenCV format
        image_data = np.asarray(bytearray(image_response.content), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Download coordinates file from S3 URL
        coords_response = requests.get(request.coordinates_file_url)
        if coords_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download coordinates file from S3")
        
        try:
            coordinates_data = coords_response.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON coordinates file")
        
        # Process the OMR sheet
        results = process_omr(image, coordinates_data)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/grade-omr/")
async def grade_omr(
    omr_image: UploadFile = File(...),
    coordinates_file: UploadFile = File(...)
):
    try:
        # Read and validate the image
        image_contents = await omr_image.read()
        nparr = np.frombuffer(image_contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Read and validate the coordinates file
        coordinates_contents = await coordinates_file.read()
        try:
            coordinates_data = json.loads(coordinates_contents)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON coordinates file")
        
        # Process the OMR sheet
        results = process_omr(image, coordinates_data)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "OMR Grading System API is running",
        "endpoints": {
            "/grade-omr/": "POST endpoint for grading OMR sheets",
            "/grade-omr-from-s3/": "POST endpoint for grading OMR sheets using S3 URLs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Use 127.0.0.1 instead of localhost and a different port
    logger.info("Starting server on http://127.0.0.1:8080")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="debug"
    )
