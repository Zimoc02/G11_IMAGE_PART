import smbus2
import time

I2C_ADDR = 0x08
bus = smbus2.SMBus(1)

def send_data():
    data = [200, 800, 1200, 1000, 0]
    byte_data = []
    
    for num in data:
        # Ensure values are within 16-bit range (0-65535)
        if num < 0 or num > 65535:
            raise ValueError(f"Value {num} out of 16-bit range")
            
        byte_data.append((num >> 8) & 0xFF)  # High byte
        byte_data.append(num & 0xFF)         # Low byte
    
    try:
        bus.write_i2c_block_data(I2C_ADDR, 0, byte_data)
        print(f"Sent: {data}")
    except Exception as e:
        print(f"I2C Error: {e}")
        # Reset I2C bus on error
        bus.close()
        time.sleep(1)
        bus = smbus2.SMBus(1)

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(2)
