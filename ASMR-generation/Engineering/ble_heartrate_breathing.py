import asyncio
from bleak import BleakClient
import time
# =========蓝牙设备配置======= #
DEVICE_MAC = "C0:00:00:00:00:30"
TX_UUID = "6e400003-b5a5-f393-e0a9-e50e24dcca9e"
RX_UUID = "6e400003-b5a5-f393-e0a9-e50e24dcca9e"
START_COMMAND = b"\x11" #0x11

#parsing for data on Bluetooth device, send heart rate & breathing rate
def parse_packet(data):
    if not len(data) > 46:
        print(f"Insufficient length for package")
        return None, None

    # check package head
    if data[0] != 0xFF or data[1] != 0x02 or data[2] != 0xFF:
        print(f"Incorrect package head")
        return None, None

    try:
        # extract heart rate
        heart_rate = data[42]
        # extract breathing rate
        breathing_rate = data[45]

        return heart_rate, breathing_rate

    except IndexError:
        print("Error parsing packet")
        return None, None







def data_handler(sender, data):
    # call package decoder function
    heart_rate, breathing_rate = parse_packet(data)

    # If parsing succeeds, print the result
    if heart_rate is not None and breathing_rate is not None:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{timestamp}]心率: {heart_rate} BPM, 呼吸率:{breathing_rate} BPM")


async def main():
    # print information
    print("Connecting to device...")
    print(f"MAC: {DEVICE_MAC}")
    print(f"TX_UUID: {TX_UUID}")
    print(f"RX_UUID: {RX_UUID}")

    try:
        async with BleakClient(DEVICE_MAC) as client:
            # check if connectoin is successful
            if not client.is_connected:
                print("Failed connection")
                return

            print("\nSuccessful connection")

            # send launch order to device
            print(f"Sending launch order: 0x{START_COMMAND.hex().upper()} ")
            await client.write_gatt_char(RX_UUID, START_COMMAND)

            # start receiving data
            print("\n Start receiving data...")
            print("-"*50)

            await client.start_notify(TX_UUID, data_handler)

            # Keep the connection and continue to receive data
            while client.is_connected:
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        print("Program stopped")
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStop")

