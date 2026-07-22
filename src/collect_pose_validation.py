import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "local_pose_validation.json"
CALIBRATION_DIR = (
    PROJECT_ROOT / "data" / "pose_validation" / "calibration"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "pose_validation" / "dataset"
)

PROTOCOL_VERSION = 2.0
ADDR_PRESENT_POSITION = 37
POSITION_UNIT_DEG = 0.29


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_calibration():
    calibration_results = load_json(
        CALIBRATION_DIR / "calibration_results.json"
    )

    camera_matrix = np.load(
        CALIBRATION_DIR
        / calibration_results["camera_matrix_file"]
    )

    distortion_coefficients = np.load(
        CALIBRATION_DIR
        / calibration_results["distortion_coefficients_file"]
    )

    board_specification = load_json(
        CALIBRATION_DIR / "board_specification.json"
    )

    return (
        camera_matrix,
        distortion_coefficients,
        board_specification,
    )


def create_charuco_board(board_specification):
    dictionary_name = board_specification["dictionary"]

    dictionary_id = getattr(cv2.aruco, dictionary_name)

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(
        dictionary_id
    )

    board = cv2.aruco.CharucoBoard(
        (
            int(board_specification["squares_x"]),
            int(board_specification["squares_y"]),
        ),
        float(board_specification["square_length_m"]),
        float(board_specification["marker_length_m"]),
        aruco_dictionary,
    )

    return board


def read_encoder_position(
    port_handler,
    packet_handler,
    motor_id,
):
    raw_position, communication_result, packet_error = (
        packet_handler.read2ByteTxRx(
            port_handler,
            motor_id,
            ADDR_PRESENT_POSITION,
        )
    )

    if communication_result != COMM_SUCCESS:
        raise RuntimeError(
            packet_handler.getTxRxResult(communication_result)
        )

    if packet_error != 0:
        raise RuntimeError(
            packet_handler.getRxPacketError(packet_error)
        )

    position_deg = raw_position * POSITION_UNIT_DEG

    return raw_position, position_deg


def estimate_charuco_pose(
    frame,
    board,
    camera_matrix,
    distortion_coefficients,
):
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_parameters.cameraMatrix = camera_matrix
    charuco_parameters.distCoeffs = distortion_coefficients

    detector_parameters = cv2.aruco.DetectorParameters()

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
    ) = detector.detectBoard(frame)

    annotated = frame.copy()

    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            annotated,
            marker_corners,
            marker_ids,
        )

    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            annotated,
            charuco_corners,
            charuco_ids,
        )

    if charuco_ids is None or len(charuco_ids) < 4:
        return None, annotated

    object_points, image_points = board.matchImagePoints(
        charuco_corners,
        charuco_ids,
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, annotated

    cv2.drawFrameAxes(
        annotated,
        camera_matrix,
        distortion_coefficients,
        rotation_vector,
        translation_vector,
        0.10,
        3,
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    camera_position_board = (
        -rotation_matrix.T @ translation_vector
    )

    return {
        "rotation_vector": rotation_vector.flatten(),
        "translation_vector": translation_vector.flatten(),
        "camera_position": camera_position_board.flatten(),
        "detected_corners": len(charuco_ids),
    }, annotated


def append_csv(csv_path, row):
    file_exists = csv_path.exists()

    fieldnames = [
        "sample_id",
        "timestamp",
        "encoder_raw",
        "encoder_deg",
        "encoder_delta_deg",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "translation_x_m",
        "translation_y_m",
        "translation_z_m",
        "rotation_vector_x",
        "rotation_vector_y",
        "rotation_vector_z",
        "detected_charuco_corners",
        "image_file",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    config = load_json(CONFIG_PATH)

    camera_index = int(config["camera_index"])
    image_width = int(config["image_width"])
    image_height = int(config["image_height"])

    motor_port = config["motor_port"]
    baud_rate = int(config["baud_rate"])
    motor_id = int(config["pan_motor_id"])

    (
        camera_matrix,
        distortion_coefficients,
        board_specification,
    ) = load_calibration()

    board = create_charuco_board(board_specification)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "pose_validation.csv"

    port_handler = PortHandler(motor_port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(
            f"Could not open motor port: {motor_port}"
        )

    if not port_handler.setBaudRate(baud_rate):
        port_handler.closePort()
        raise RuntimeError(
            f"Could not set baud rate: {baud_rate}"
        )

    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_AVFOUNDATION,
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    if not cap.isOpened():
        port_handler.closePort()
        raise RuntimeError(
            f"Could not open camera index {camera_index}"
        )

    initial_encoder_deg = None
    sample_id = 0

    print("Pose validation collector")
    print("-------------------------")
    print("Move Gretchen manually to each desired position.")
    print("Keep the ChArUco board stationary.")
    print("Press SPACE to save a sample.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read camera frame.")
                continue

            pose, annotated = estimate_charuco_pose(
                frame,
                board,
                camera_matrix,
                distortion_coefficients,
            )

            encoder_raw, encoder_deg = read_encoder_position(
                port_handler,
                packet_handler,
                motor_id,
            )

            if initial_encoder_deg is None:
                initial_encoder_deg = encoder_deg

            encoder_delta_deg = (
                encoder_deg - initial_encoder_deg
            )

            cv2.putText(
                annotated,
                f"Encoder: {encoder_deg:.2f} deg",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Delta: {encoder_delta_deg:.2f} deg",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if pose is None:
                status_text = "ChArUco pose unavailable"
            else:
                status_text = (
                    f"Pose OK, corners: "
                    f"{pose['detected_corners']}"
                )

            cv2.putText(
                annotated,
                status_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0)
                if pose is not None
                else (0, 0, 255),
                2,
            )

            cv2.imshow(
                "Gretchen Pose Validation",
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                if pose is None:
                    print(
                        "Sample not saved: "
                        "ChArUco pose was not detected."
                    )
                    continue

                timestamp = datetime.now().isoformat(
                    timespec="seconds"
                )

                image_name = (
                    f"sample_{sample_id:03d}.png"
                )

                image_path = images_dir / image_name

                cv2.imwrite(
                    str(image_path),
                    annotated,
                )

                camera_position = pose["camera_position"]
                translation = pose["translation_vector"]
                rotation_vector = pose["rotation_vector"]

                row = {
                    "sample_id": sample_id,
                    "timestamp": timestamp,
                    "encoder_raw": encoder_raw,
                    "encoder_deg": encoder_deg,
                    "encoder_delta_deg": (
                        encoder_delta_deg
                    ),
                    "camera_x_m": camera_position[0],
                    "camera_y_m": camera_position[1],
                    "camera_z_m": camera_position[2],
                    "translation_x_m": translation[0],
                    "translation_y_m": translation[1],
                    "translation_z_m": translation[2],
                    "rotation_vector_x": rotation_vector[0],
                    "rotation_vector_y": rotation_vector[1],
                    "rotation_vector_z": rotation_vector[2],
                    "detected_charuco_corners": (
                        pose["detected_corners"]
                    ),
                    "image_file": str(image_path),
                }

                append_csv(csv_path, row)

                print(
                    f"Saved sample {sample_id}: "
                    f"encoder={encoder_deg:.2f}°, "
                    f"delta={encoder_delta_deg:.2f}°"
                )

                sample_id += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        port_handler.closePort()

        print(f"Dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()

