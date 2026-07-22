from dynamixel_sdk import *

DEVICE="/dev/cu.usbserial-FT94EP35"
BAUD=57600

port=PortHandler(DEVICE)
packet=PacketHandler(2.0)

port.openPort()
port.setBaudRate(BAUD)

print("Scanning IDs 0~20...\n")

for dxl_id in range(21):
    model, result, error = packet.ping(port, dxl_id)

    if result == COMM_SUCCESS and error == 0:
        print(f"FOUND ID {dxl_id}   model {model}")

port.closePort()
