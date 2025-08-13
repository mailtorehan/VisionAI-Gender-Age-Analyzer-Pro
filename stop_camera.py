import tkinter as tk
from tkinter import ttk

def show_hello_page():
    root = tk.Tk()
    root.title("Stop Camera")
    root.geometry("300x200")
    
    label = ttk.Label(root, text="Hello from Stop Camera!", font=('Arial', 14))
    label.pack(pady=80)
    
    root.mainloop()

if __name__ == "__main__":
    show_hello_page()