import cv2
import cv2.aruco as aruco

# Create dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generate markers
for id in range(1, 5):
    marker = aruco.generateImageMarker(aruco_dict, id, 200)  # 200x200 px
    cv2.imwrite(f"marker_{id}.png", marker)