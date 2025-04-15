import tkinter as tk
import subprocess
import os
import signal

# 全局变量
process = None

def start_script():
    global process
    if process is None:
        process = subprocess.Popen(
            ['python3', 'cubic2_aruco_3_Homography1.py'],
            stdin=subprocess.PIPE  # ✅ 关键！
        )
        status_label.config(text="✅ 程序已启动", fg="green")
    else:
        status_label.config(text="⚠️ 程序已在运行", fg="orange")

def stop_script():
    global process
    if process is not None:
        os.kill(process.pid, signal.SIGTERM)
        process = None
        status_label.config(text="🛑 程序已终止", fg="red")
    else:
        status_label.config(text="⚠️ 没有程序在运行", fg="gray")

def send_key_to_script(key):
    global process
    if process is not None and process.stdin is not None:
        try:
            process.stdin.write(f"{key}\n".encode())
            process.stdin.flush()
            status_label.config(text=f"📤 已发送按键: {key}", fg="blue")
        except Exception as e:
            status_label.config(text=f"❌ 发送失败: {e}", fg="red")
    else:
        status_label.config(text="⚠️ 程序未启动，无法发送按键", fg="orange")

def regenerate_path():
    send_key_to_script('p')

def save_accuracy():
    send_key_to_script('s')

# GUI 主窗口
root = tk.Tk()
root.title("红球控制 GUI")
root.geometry("300x300")

tk.Button(root, text="▶️ Start", command=start_script, width=20, height=2, bg="lightgreen").pack(pady=5)
tk.Button(root, text="⏹️ Stop", command=stop_script, width=20, height=2, bg="salmon").pack(pady=5)
tk.Button(root, text="🔁 Re-generate Path", command=regenerate_path, width=20, height=2).pack(pady=5)
tk.Button(root, text="💾 Save Accuracy", command=save_accuracy, width=20, height=2).pack(pady=5)

status_label = tk.Label(root, text="等待启动...", font=("Arial", 11), fg="black")
status_label.pack(pady=10)

root.mainloop()
