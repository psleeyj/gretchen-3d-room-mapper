import json
from pathlib import Path

from dynamixel_sdk import (
    COMM_SUCCESS,
    PacketHandler,
    PortHandler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "local_pose_validation.json"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def try_protocol(port_handler, motor_ids, protocol_version):
    packet_handler = PacketHandler(protocol_version)

    print(f"\nTrying protocol {protocol_version}...")

    found = []

    for motor_id in motor_ids:
        model_number, communication_result, error = packet_handler.ping(
            port_handler,
            motor_id,
        )

        if communication_result == COMM_SUCCESS and error == 0:
            print(
                f"FOUND motor ID {motor_id}: "
                f"model number {model_number}"
            )
            found.append(
                {
                    "id": motor_id,
                    "model_number": model_number,
                    "protocol": protocol_version,
                }
            )
        else:
            communication_message = packet_handler.getTxRxResult(
                communication_result
            )
            error_message = packet_handler.getRxPacketError(error)

            print(
                f"No response from ID {motor_id}: "
                f"{communication_message}"
            )

            if error:
                print(f"  Packet error: {error_message}")

    return found


def main():
    config = load_config()

    device_name = config["motor_port"]
    baud_rate = int(config["baud_rate"])

    motor_ids = [
        int(config["pan_motor_id"]),
        int(config["tilt_motor_id"]),
    ]

    print(f"Opening port: {device_name}")

    port_handler = PortHandler(device_name)

    if not port_handler.openPort():
        raise RuntimeError(f"Could not open port: {device_name}")

    try:
        if not port_handler.setBaudRate(baud_rate):
            raise RuntimeError(
                f"Could not set baud rate to {baud_rate}"
            )

        print(f"Baud rate set to: {baud_rate}")
        print(f"Testing motor IDs: {motor_ids}")
        print("This script only sends ping commands.")
        print("It does not move the motors.")

        all_found = []

        for protocol_version in (1.0, 2.0):
            found = try_protocol(
                port_handler,
                motor_ids,
                protocol_version,
            )
            all_found.extend(found)

        print("\nSummary:")

        if not all_found:
            print("No motors responded.")
        else:
            for motor in all_found:
                print(
                    f"ID {motor['id']} | "
                    f"model {motor['model_number']} | "
                    f"protocol {motor['protocol']}"
                )

    finally:
        port_handler.closePort()
        print("\nSerial port closed.")


if __name__ == "__main__":
    main()
