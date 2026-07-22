import json
from pathlib import Path

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "local_pose_validation.json"

PROTOCOL_VERSION = 2.0

# XL-320 control table
ADDR_PRESENT_POSITION = 37
PRESENT_POSITION_SIZE = 2

POSITION_UNIT_DEG = 0.29


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    port_name = config["motor_port"]
    baud_rate = int(config["baud_rate"])
    motor_id = int(config["pan_motor_id"])

    port_handler = PortHandler(port_name)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        raise RuntimeError(f"Could not open serial port: {port_name}")

    try:
        if not port_handler.setBaudRate(baud_rate):
            raise RuntimeError(
                f"Could not set baud rate to {baud_rate}"
            )

        position_raw, communication_result, packet_error = (
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

        position_deg = position_raw * POSITION_UNIT_DEG

        print(f"Motor ID: {motor_id}")
        print(f"Raw position: {position_raw}")
        print(f"Position: {position_deg:.2f} degrees")

    finally:
        port_handler.closePort()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
