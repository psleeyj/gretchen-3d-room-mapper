import cv2
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "hardware.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

camera_index = config["camera"]["index"]
width = config["camera"].get("width", 640)
height = config["camera"].get("height", 480)

save_dir = ROOT / "data" / "calibration" / "raw_images"
save_dir.mkdir(parents=True, exist_ok=True)

existing_images = list(save_dir.glob("image_*.png"))
image_number = len(existing_images) + 1

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

import time
time.sleep(2)

# Use the camera's native resolution.
# Forcing width/height caused frame capture failures on this device.
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {camera_index}")

print("SPACE: save image")
print("Q: quit")
print(f"Saving images to: {save_dir}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        continue

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    marker_count = 0

    if marker_ids is not None:
        marker_count = len(marker_ids)
        cv2.aruco.drawDetectedMarkers(
            display,
            marker_corners,
            marker_ids
        )

    saved_count = image_number - 1

    cv2.putText(
        display,
        f"Markers: {marker_count}/17",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Saved: {saved_count}/30",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        "SPACE = save | Q = quit",
        (20, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Gretchen Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        if marker_count < 4:
            print("Not saved: show at least 4 markers.")
            continue

        filename = save_dir / f"image_{image_number:03d}.png"

        if cv2.imwrite(str(filename), frame):
            print(
                f"Saved {filename.name} "
                f"with {marker_count} detected markers."
            )
            image_number += 1
        else:
            print(f"Failed to save {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Finished. Total saved images: {image_number - 1}")
