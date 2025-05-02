import cv2
import json
import numpy as np

# === Config ===
IMAGE_PATH = "output/detect_omr/warped_omr.jpg"
JSON_PATH = "assets/omr_coordinates.json"
OUTPUT_PATH = "output/verify_coordinates/omr_marked_preview.jpg"

# === Load Image ===
img = cv2.imread(IMAGE_PATH)
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
draw_bubbles(data.get("student_id", []), (0, 255, 255), "ID")

# Exam Date Bubbles
draw_bubbles(data.get("exam_date", []), (255, 255, 0), "D")

# Questions
for idx, (q, bubbles) in enumerate(data.get("questions", {}).items()):
    draw_bubbles(bubbles, (0, 255, 0), f"Q{idx+1}_")

# Save output
cv2.imwrite(OUTPUT_PATH, annotated_img)
print("✅ Done! Annotated image saved as:", OUTPUT_PATH)
