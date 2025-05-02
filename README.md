# OMR (Optical Mark Recognition) Grading System

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

### 3. Using the Bubble Mapping Tool

The project includes an interactive web-based tool (`omr_bubble_mapper.html`) for mapping bubble locations on your answer sheet:

1. Open `omr_bubble_mapper.html` in your web browser
2. Upload your processed answer sheet image
3. Use the tool to mark bubble locations:
   - Select the section type (Student ID or Questions)
   - Choose between point or circle drawing mode
   - Adjust circle size if needed
   - Click on each bubble location to mark it
4. Download the coordinates as JSON for use in your grading system

Features of the Bubble Mapper:
- Support for Student ID and Questions sections
- Configurable number of options per question
- Real-time coordinate display
- Circle or point marking modes
- Adjustable circle size
- JSON export of coordinates

### Output Files

The script generates several output files in the `output` directory:
- `detected_markers.jpg`: Shows detected markers with IDs and corners
- `warped_omr.jpg`: The corrected and aligned answer sheet
- `warped_omr_resized.jpg`: A resized version if the original is too large
- `partial_detection.jpg`: Debug image when not all markers are detected

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