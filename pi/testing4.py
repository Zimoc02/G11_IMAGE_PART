import smbus2
import time

I2C_ADDR = 0x08  # Arduino I2C address

# Initialize bus globally
bus = smbus2.SMBus(1)

def send_data():
    global bus  # Declare bus as global
    data = [200, 800, 1200, 1000, 0]
    byte_data = []
    
    for num in data:
        if num < 0 or num > 65535:
            raise ValueError(f"Value {num} out of 16-bit range")
            
        byte_data.append((num >> 8) & 0xFF)
        byte_data.append(num & 0xFF)
    
    try:
        bus.write_i2c_block_data(I2C_ADDR, 0, byte_data)
        print(f"Sent: {data}")
    except Exception as e:
        print(f"I2C Error: {e}")
        # Properly handle bus reset
        try:
            bus.close()
        except:
            pass
        # Reinitialize globally
        bus = smbus2.SMBus(1)
        time.sleep(1)

if __name__ == "__main__":
    while True:
        send_data()
        time.sleep(2)
