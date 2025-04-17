import smbus
import time

I2C_ADDR = 0x08  # Arduino I2C address
bus = smbus.SMBus(1)  # Use I2C-1 on Raspberry Pi

data = [2, 3, 4, 5, 0]  # Data to send

def send_data():
    try:
        bus.write_i2c_block_data(I2C_ADDR, 0, data)  # Send data block
        print("Data sent:", data)
    except Exception as e:
        print("Error:", e)

while True:
    send_data()
    time.sleep(1)  # Send data every second
