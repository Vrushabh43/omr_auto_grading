import cv2
import json
import numpy as np
import os

os.makedirs('output', exist_ok=True)
os.makedirs('output/verify_coordinates', exist_ok=True)
# Remove all files from output/verify_coordinates directory
if os.path.exists('output/verify_coordinates'):
    for file in os.listdir('output/verify_coordinates'):
        file_path = os.path.join('output/verify_coordinates', file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# === Config ===
IMAGE_PATH = "output/detect_omr/warped_omr_resized.jpg"
JSON_PATH = "assets/omr_coordinates.json"
OUTPUT_PATH = "output/verify_coordinates/omr_marked_preview.jpg"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load Image ===
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Input image not found at {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Failed to load image from {IMAGE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/verify_coordinates/step1_gray.jpg", gray)

# === Apply Threshold for binary view ===
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("output/verify_coordinates/step2_threshold.jpg", thresh)

# === Load coordinates ===
with open(JSON_PATH, "r") as f:
    data = json.load(f)

annotated_img = img.copy()

# === Draw circles ===
def draw_bubbles(bubble_list, color, label_prefix=None):
    for i, bubble in enumerate(bubble_list):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        cv2.circle(annotated_img, (x, y), r, color, 2)
        if label_prefix:
            cv2.putText(annotated_img, f"{label_prefix}{i+1}", (x - r, y - r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Student ID Bubbles
for letter_key, bubbles in data["student_id"].items():
    draw_bubbles(bubbles, (255, 0, 0), f"ID{letter_key[-1]}")

# Paper Code Bubbles
for letter_key, bubbles in data["paper_code"].items():
    draw_bubbles(bubbles, (255, 255, 0), f"PC{letter_key[-1]}")

# Questions
for q_key, bubbles in data["questions"].items():
    q_num = int(q_key.split('_')[1])
    draw_bubbles(bubbles, (0, 255, 0), f"Q{q_num}_")

# Save output
cv2.imwrite(OUTPUT_PATH, annotated_img)
print("✅ Done! Annotated image saved as:", OUTPUT_PATH)
