import smbus2
import time

I2C_ADDR = 0x08  # Arduino I2C address
bus = smbus2.SMBus(1)  # Use I2C bus 1

data = [200, 800, 1200, 1000, 0]

def send_data():
    byte_data = []
    for num in data:
        byte_data.append(num >> 8)   # High byte
        byte_data.append(num & 0xFF) # Low byte
    
    try:
        bus.write_i2c_block_data(I2C_ADDR, 0, byte_data)
        print("Data sent:", data)
    except Exception as e:
        print("I2C Error:", e)

while True:
    send_data()
    time.sleep(2)  # Send data every 2 seconds
