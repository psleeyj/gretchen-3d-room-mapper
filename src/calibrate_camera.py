import cv2
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "calibration" / "raw_images"
RESULT_DIR = ROOT / "data" / "calibration" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SQUARES_X = 7
SQUARES_Y = 5

# meters
SQUARE_LENGTH = 0.0465
MARKER_LENGTH = 0.0300

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

detector_parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(
    dictionary,
    detector_parameters
)

all_charuco_corners = []
all_charuco_ids = []

image_size = None
used_images = []
rejected_images = []

image_paths = sorted(IMAGE_DIR.glob("*.png"))

if not image_paths:
    raise RuntimeError(
        f"No calibration images found in {IMAGE_DIR}"
    )

print(f"Found {len(image_paths)} images.")

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        rejected_images.append(image_path.name)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    marker_corners, marker_ids, _ = (
        aruco_detector.detectMarkers(gray)
    )

    if marker_ids is None or len(marker_ids) < 4:
        print(f"Rejected {image_path.name}: not enough markers.")
        rejected_images.append(image_path.name)
        continue

    response, charuco_corners, charuco_ids = (
        cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board
        )
    )

    if (
        charuco_corners is None
        or charuco_ids is None
        or len(charuco_ids) < 6
    ):
        print(
            f"Rejected {image_path.name}: "
            "not enough ChArUco corners."
        )
        rejected_images.append(image_path.name)
        continue

    all_charuco_corners.append(charuco_corners)
    all_charuco_ids.append(charuco_ids)
    used_images.append(image_path.name)

    print(
        f"Accepted {image_path.name}: "
        f"{len(marker_ids)} markers, "
        f"{len(charuco_ids)} corners."
    )

if len(all_charuco_corners) < 10:
    raise RuntimeError(
        f"Only {len(all_charuco_corners)} usable images. "
        "Capture at least 10 good images, preferably 20–30."
    )

calibration_result = cv2.aruco.calibrateCameraCharucoExtended(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

(
    reprojection_error,
    camera_matrix,
    distortion_coefficients,
    rotation_vectors,
    translation_vectors,
    intrinsic_std_deviations,
    extrinsic_std_deviations,
    per_view_errors
) = calibration_result

np.save(
    RESULT_DIR / "camera_matrix.npy",
    camera_matrix
)

np.save(
    RESULT_DIR / "distortion_coefficients.npy",
    distortion_coefficients
)

np.save(
    RESULT_DIR / "per_view_errors.npy",
    per_view_errors
)

result_data = {
    "opencv_version": cv2.__version__,
    "image_width": int(image_size[0]),
    "image_height": int(image_size[1]),
    "squares_x": SQUARES_X,
    "squares_y": SQUARES_Y,
    "square_length_m": SQUARE_LENGTH,
    "marker_length_m": MARKER_LENGTH,
    "dictionary": "DICT_6X6_250",
    "number_of_images_found": len(image_paths),
    "number_of_images_used": len(used_images),
    "reprojection_error": float(reprojection_error),
    "camera_matrix": camera_matrix.tolist(),
    "distortion_coefficients": (
        distortion_coefficients.tolist()
    ),
    "used_images": used_images,
    "rejected_images": rejected_images
}

with open(
    RESULT_DIR / "calibration_results.yaml",
    "w"
) as f:
    yaml.safe_dump(
        result_data,
        f,
        sort_keys=False
    )

print()
print("Calibration complete.")
print(f"Images used: {len(used_images)}")
print(f"Images rejected: {len(rejected_images)}")
print(f"Reprojection error: {reprojection_error:.4f}")
print()
print("Camera matrix:")
print(camera_matrix)
print()
print("Distortion coefficients:")
print(distortion_coefficients)
print()
print(f"Results saved to: {RESULT_DIR}")
