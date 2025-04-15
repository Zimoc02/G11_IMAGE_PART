import tkinter as tk
import subprocess
import os
import signal

# 全局变量用于保存子进程
process = None

# 启动 xx.py
def start_script():
    global process
    if process is None:
        process = subprocess.Popen(['python3', 'xx.py'])
        status_label.config(text="程序运行中", fg="green")

# 停止 xx.py
def stop_script():
    global process
    if process is not None:
        os.kill(process.pid, signal.SIGTERM)
        process = None
        status_label.config(text="程序已停止", fg="red")

# 创建 GUI 界面
root = tk.Tk()
root.title("红球路径控制 GUI")
root.geometry("300x180")

start_btn = tk.Button(root, text="Start", command=start_script, bg="lightgreen", width=15, height=2)
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="End", command=stop_script, bg="salmon", width=15, height=2)
stop_btn.pack(pady=10)

status_label = tk.Label(root, text="等待启动...", font=("Arial", 12))
status_label.pack()

root.mainloop()
