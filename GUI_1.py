import tkinter as tk
import subprocess
import os
import signal
import time

process = None

def start_script():
    global process
    if process is None:
        process = subprocess.Popen(['python3', 'cubic2_aruco_3_Homography1.py'])
        status_label.config(text="程序运行中", fg="green")

def stop_script():
    global process
    if process is not None:
        os.kill(process.pid, signal.SIGTERM)
        process = None
        status_label.config(text="程序已停止", fg="red")

def send_key_to_script(key):
    global process
    if process is not None:
        try:
            process.stdin.write(f"{key}\n".encode())
            process.stdin.flush()
        except Exception as e:
            print(f"⚠️ 无法发送按键: {e}")

def regenerate_path():
    print("🌀 发送热键 p（重新生成路径）")
    send_key_to_script('p')

def save_accuracy():
    print("💾 发送热键 s（保存误差数据）")
    send_key_to_script('s')

# 主窗口
root = tk.Tk()
root.title("红球控制 GUI")
root.geometry("320x300")

tk.Button(root, text="▶️ Start", command=start_script, width=25, height=2, bg="lightgreen").pack(pady=8)
tk.Button(root, text="⏹️ End", command=stop_script, width=25, height=2, bg="salmon").pack(pady=8)
tk.Button(root, text="🔁 Re-generate Path", command=regenerate_path, width=25, height=2).pack(pady=8)
tk.Button(root, text="💾 Save Accuracy", command=save_accuracy, width=25, height=2).pack(pady=8)

status_label = tk.Label(root, text="等待启动...", fg="blue", font=("Arial", 12))
status_label.pack(pady=10)

root.mainloop()
