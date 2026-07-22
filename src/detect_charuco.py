import cv2
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "config" / "hardware.yaml") as f:
    config = yaml.safe_load(f)

camera_index = config["camera"]["index"]

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

print("Camera:", camera_index)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    display = frame.copy()

    if ids is not None:
        print("Detected markers:", len(ids))
        cv2.aruco.drawDetectedMarkers(display, corners, ids)

    cv2.imshow("Debug", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
