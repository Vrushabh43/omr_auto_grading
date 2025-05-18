# OMR (Optical Mark Recognition) Grading System

> **General Instruction:**
> This repository provides a complete pipeline for automated OMR grading using Python. To get started, follow the installation steps, then use the provided scripts in the order described below. Each script has a specific role in the workflow. See the "Python Scripts Overview" section for details.

This project implements an automated OMR (Optical Mark Recognition) system for grading answer sheets using ArUco markers for precise detection and alignment.

## Features

- Automatic detection of ArUco markers for sheet alignment
- Perspective transformation for correcting skewed images
- High-precision bubble detection and grading
- Support for multiple answer sheet formats
- Debug visualization of marker detection
- Interactive bubble mapping tool for custom answer sheet layouts

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy
- imutils
- Modern web browser (for bubble mapping tool)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd omr_grading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install opencv-python numpy imutils
```

## Usage

### 1. Generate ArUco Markers

First, generate the required ArUco markers for your answer sheet:
```bash
python aruco_gen.py
```
This will create four markers (marker_1.png through marker_4.png) that should be placed at the corners of your answer sheet.

### 2. Process Answer Sheets

Place your answer sheet image in the `assets` directory and run:
```bash
python detect_omr.py
```

The script will:
- Detect the ArUco markers
- Apply perspective transformation
- Save the processed image in the `output` directory

### 3. Generate Bubble Coordinates (Choose One Method)

You need a JSON file with bubble coordinates for grading. You can generate this in one of two ways:

**A. Using the Python script:**
```bash
python generate_omr_coordinates.py
```
This will create `assets/omr_coordinates.json` with default coordinates for Student ID, Paper Code, and Questions.

**B. Using the interactive HTML tool:**
1. Open `omr_bubble_mapper.html` in your web browser.
2. Upload your processed answer sheet image.
3. Mark bubble locations as needed and export the coordinates as JSON.
4. Save the exported JSON as `assets/omr_coordinates.json`.

### 4. (Optional) Verify Bubble Coordinates

To visually verify that your coordinates align with the answer sheet:
```bash
python verify_coordinates.py
```
This will overlay the coordinates on the processed sheet and save a preview image in `output/verify_coordinates/omr_marked_preview.jpg`.

### 5. Grade the Answer Sheet

Run the grading script to detect filled bubbles and output results:
```bash
python omr_grader.py
```

### Output Files

The script generates several output files in the `output` directory:
- `detected_markers.jpg`: Shows detected markers with IDs and corners
- `warped_omr.jpg`: The corrected and aligned answer sheet
- `warped_omr_resized.jpg`: A resized version if the original is too large
- `partial_detection.jpg`: Debug image when not all markers are detected

## Python Scripts Overview

Below is a summary of each Python script in this repository and how to run them:

- **aruco_gen.py**: Generates four ArUco marker images (marker_1.png to marker_4.png) for use on the answer sheet corners.
  - **Run:** `python aruco_gen.py`

- **detect_omr.py**: Detects ArUco markers in a scanned answer sheet, applies perspective correction, and outputs aligned images.
  - **Run:** `python detect_omr.py`

- **generate_omr_coordinates.py**: Generates a JSON file (`omr_coordinates.json`) with the coordinates of all bubbles (Student ID, Paper Code, Questions) for grading.
  - **Run:** `python generate_omr_coordinates.py`

- **verify_coordinates.py**: Overlays and visualizes the bubble coordinates on the processed answer sheet for verification.
  - **Run:** `python verify_coordinates.py`

- **omr_grader.py**: Grades the answer sheet by detecting filled bubbles using the processed image and coordinates JSON, and outputs results.
  - **Run:** `python omr_grader.py`

## Marker Placement

For optimal results, place the ArUco markers at the four corners of your answer sheet:
- Marker 1: Top-left corner
- Marker 2: Top-right corner
- Marker 3: Bottom-left corner
- Marker 4: Bottom-right corner

## Troubleshooting

- Ensure good lighting conditions when scanning answer sheets
- Make sure all four markers are visible and not obstructed
- Check that the markers are properly aligned with the corners
- If detection fails, check the debug images in the output directory
- When using the bubble mapper, ensure your image is properly aligned and markers are visible

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable] 