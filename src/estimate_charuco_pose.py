import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "pose_validation"
    / "test_images"
    / "charuco_test.png"
)

DEFAULT_CALIBRATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "pose_validation"
    / "calibration"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pose_validation"
    / "outputs"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate camera pose from a ChArUco board image."
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input ChArUco image.",
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing camera calibration and board files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where results will be saved.",
    )

    return parser.parse_args()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_calibration(calibration_dir):
    calibration_results_path = calibration_dir / "calibration_results.json"
    calibration_results = load_json(calibration_results_path)

    camera_matrix_path = (
        calibration_dir / calibration_results["camera_matrix_file"]
    )

    distortion_path = (
        calibration_dir
        / calibration_results["distortion_coefficients_file"]
    )

    if not camera_matrix_path.exists():
        raise FileNotFoundError(
            f"Camera matrix not found: {camera_matrix_path}"
        )

    if not distortion_path.exists():
        raise FileNotFoundError(
            f"Distortion coefficients not found: {distortion_path}"
        )

    camera_matrix = np.load(camera_matrix_path)
    distortion_coefficients = np.load(distortion_path)

    return camera_matrix, distortion_coefficients, calibration_results


def create_charuco_board(board_specification):
    dictionary_name = board_specification["dictionary"]

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(
            f"Unsupported ArUco dictionary: {dictionary_name}"
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    squares_x = int(board_specification["squares_x"])
    squares_y = int(board_specification["squares_y"])
    square_length = float(board_specification["square_length_m"])
    marker_length = float(board_specification["marker_length_m"])

    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dictionary,
    )

    return board, aruco_dictionary


def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(
        rotation_matrix[0, 0] ** 2
        + rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2],
        )

        y = np.arctan2(
            -rotation_matrix[2, 0],
            sy,
        )

        z = np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0],
        )

    else:
        x = np.arctan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1],
        )

        y = np.arctan2(
            -rotation_matrix[2, 0],
            sy,
        )

        z = 0.0

    return np.degrees(np.array([x, y, z]))


def calculate_reprojection_error(
    object_points,
    image_points,
    rotation_vector,
    translation_vector,
    camera_matrix,
    distortion_coefficients,
):
    projected_points, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )

    projected_points = projected_points.reshape(-1, 2)
    observed_points = image_points.reshape(-1, 2)

    errors = np.linalg.norm(
        observed_points - projected_points,
        axis=1,
    )

    return float(np.mean(errors))


def estimate_pose(
    image,
    board,
    aruco_dictionary,
    camera_matrix,
    distortion_coefficients,
):
    detector_parameters = cv2.aruco.DetectorParameters()

    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_parameters.cameraMatrix = camera_matrix
    charuco_parameters.distCoeffs = distortion_coefficients

    detector = cv2.aruco.CharucoDetector(
        board,
        charuco_parameters,
        detector_parameters,
    )

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = detector.detectBoard(image)

    annotated_image = image.copy()

    marker_count = 0
    charuco_corner_count = 0

    if marker_ids is not None and len(marker_ids) > 0:
        marker_count = len(marker_ids)

        cv2.aruco.drawDetectedMarkers(
            annotated_image,
            marker_corners,
            marker_ids,
        )

    if charuco_ids is not None and len(charuco_ids) > 0:
        charuco_corner_count = len(charuco_ids)

        cv2.aruco.drawDetectedCornersCharuco(
            annotated_image,
            charuco_corners,
            charuco_ids,
        )

    print(f"Detected markers: {marker_count}")
    print(f"Detected ChArUco corners: {charuco_corner_count}")

    if charuco_ids is None or len(charuco_ids) < 4:
        return None, annotated_image

    object_points, image_points = board.matchImagePoints(
        charuco_corners,
        charuco_ids,
    )

    if object_points is None or len(object_points) < 4:
        return None, annotated_image

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, annotated_image

    cv2.drawFrameAxes(
        annotated_image,
        camera_matrix,
        distortion_coefficients,
        rotation_vector,
        translation_vector,
        0.10,
        3,
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

    reprojection_error = calculate_reprojection_error(
        object_points,
        image_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )

    camera_rotation_matrix = rotation_matrix.T

    camera_position_board = (
        -camera_rotation_matrix @ translation_vector
    )

    pose_result = {
        "detected_markers": int(marker_count),
        "detected_charuco_corners": int(charuco_corner_count),
        "rotation_vector_board_to_camera": (
            rotation_vector.flatten().tolist()
        ),
        "translation_vector_board_to_camera_m": (
            translation_vector.flatten().tolist()
        ),
        "rotation_matrix_board_to_camera": (
            rotation_matrix.tolist()
        ),
        "euler_angles_board_to_camera_deg": {
            "roll_x": float(euler_angles[0]),
            "pitch_y": float(euler_angles[1]),
            "yaw_z": float(euler_angles[2]),
        },
        "camera_position_in_board_coordinates_m": (
            camera_position_board.flatten().tolist()
        ),
        "camera_rotation_matrix_in_board_coordinates": (
            camera_rotation_matrix.tolist()
        ),
        "reprojection_error_px": float(reprojection_error),
    }

    return pose_result, annotated_image


def save_results(
    output_dir,
    image_path,
    annotated_image,
    pose_result,
    calibration_results,
    board_specification,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "charuco_pose_annotated.png"
    json_path = output_dir / "charuco_pose_result.json"

    cv2.imwrite(str(annotated_path), annotated_image)

    full_result = {
        "input_image": str(image_path),
        "calibration": calibration_results,
        "board": board_specification,
        "pose": pose_result,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(full_result, file, indent=2)

    print(f"Annotated image saved to: {annotated_path}")
    print(f"Pose result saved to: {json_path}")


def main():
    args = parse_arguments()

    image_path = args.image.expanduser().resolve()
    calibration_dir = args.calibration_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    camera_matrix, distortion_coefficients, calibration_results = (
        load_calibration(calibration_dir)
    )

    board_specification = load_json(
        calibration_dir / "board_specification.json"
    )

    board, aruco_dictionary = create_charuco_board(
        board_specification
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read image: {image_path}"
        )

    print("Camera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(distortion_coefficients)

    pose_result, annotated_image = estimate_pose(
        image,
        board,
        aruco_dictionary,
        camera_matrix,
        distortion_coefficients,
    )

    if pose_result is None:
        output_dir.mkdir(parents=True, exist_ok=True)

        failed_image_path = (
            output_dir / "charuco_detection_failed.png"
        )

        cv2.imwrite(str(failed_image_path), annotated_image)

        print("\nPose estimation failed.")
        print("At least four valid ChArUco corners are required.")
        print(f"Detection image saved to: {failed_image_path}")
        return

    print("\nPose estimation successful.")

    print("\nBoard-to-camera translation in meters:")
    print(
        np.array(
            pose_result["translation_vector_board_to_camera_m"]
        )
    )

    print("\nCamera position in board coordinates in meters:")
    print(
        np.array(
            pose_result["camera_position_in_board_coordinates_m"]
        )
    )

    print("\nBoard-to-camera Euler angles in degrees:")
    print(
        pose_result["euler_angles_board_to_camera_deg"]
    )

    print(
        "\nMean reprojection error: "
        f"{pose_result['reprojection_error_px']:.4f} pixels"
    )

    save_results(
        output_dir,
        image_path,
        annotated_image,
        pose_result,
        calibration_results,
        board_specification,
    )

    cv2.imshow("ChArUco Pose Estimation", annotated_image)

    print("\nPress any key inside the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
