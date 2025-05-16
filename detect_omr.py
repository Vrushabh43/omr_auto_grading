import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.perspective import four_point_transform
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Read image
image = cv2.imread("assets/scanned_omr.png")
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
            
            # Add minimal padding to the warping points to avoid cutting edges
            ordered_pts[0] = ordered_pts[0] - [pad_x, pad_y]  # Top-left
            ordered_pts[1] = ordered_pts[1] + [pad_x, -pad_y]  # Top-right
            ordered_pts[2] = ordered_pts[2] + [pad_x, pad_y]   # Bottom-right
            ordered_pts[3] = ordered_pts[3] + [-pad_x, pad_y]  # Bottom-left
            
            # Warp the image
            warped = four_point_transform(image, ordered_pts)
            
            # Save both original size and a resized version
            cv2.imwrite("output/detect_omr/warped_omr.jpg", warped)
            
            # Save a resized version if the warped image is too large
            max_dimension = 1500
            h, w = warped.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                warped_resized = cv2.resize(warped, new_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite("output/detect_omr/warped_omr_resized.jpg", warped_resized)
                print("Saved both original and resized warped images!")
            else:
                print("Successfully warped and saved the image!")
            
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
