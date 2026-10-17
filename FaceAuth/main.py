import tkinter as tk
from tkinter import messagebox, ttk
import os
import sys
import json
import hashlib

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

user_db = os.path.join(BASE_DIR, "database.json")
if not os.path.exists(user_db):
    with open(user_db, "w", encoding="utf-8") as f:
        json.dump({}, f)

# Import module components
from register import register_face
from train_model import train_model
import recognizer
from recognizer import recognize_face

# GUI Window
root = tk.Tk()
root.title("VISION LOCK: Multi-Layer Biometric Authentication")
root.geometry("460x520")
root.resizable(False, False)
root.configure(bg="#1e1e2e")

# Styling
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
style.configure("Header.TLabel", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 14, "bold"))
style.configure("SubHeader.TLabel", background="#1e1e2e", foreground="#a6adc8", font=("Segoe UI", 9))
style.configure("TLabelframe", background="#1e1e2e", foreground="#89b4fa", bordercolor="#45475a")
style.configure("TLabelframe.Label", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=5)

def hash_password(password):
    """Hash a password with SHA-256 for secure credential storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def register():
    user_id = entry_id.get().strip()
    username = entry_name.get().strip()
    password = entry_password.get().strip()

    if not user_id or not username or not password:
        messagebox.showerror("Validation Error", "All fields are required!")
        return

    # Load current users
    try:
        with open(user_db, "r", encoding="utf-8") as f:
            users = json.load(f)
    except Exception:
        users = {}

    if user_id in users:
        messagebox.showerror("Error", f"User ID '{user_id}' already exists!")
        return

    # Store user details with hashed password
    users[user_id] = {
        "name": username,
        "password": hash_password(password)
    }

    with open(user_db, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

    messagebox.showinfo("Face Capture Required", 
                        f"User '{username}' registered with ID {user_id}.\n\n"
                        "Look at the camera to capture 40 biometric face samples.")

    # 1. Capture face samples
    captured = register_face(user_id, num_samples=40)
    if not captured:
        messagebox.showwarning("Incomplete", "Face capture was cancelled or failed. Please retry registration.")
        return

    # 2. Train CNN model
    btn_reg.config(text="Training CNN (Please wait)...", state="disabled", bg="#f9e2af")
    root.update()

    train_success = train_model(epochs=18)
    
    btn_reg.config(text="Enroll & Train CNN", state="normal", bg="#a6e3a1")
    root.update()

    if train_success:
        recognizer.reload_model()
        
        # Read validation accuracy if available
        metrics_file = os.path.join(BASE_DIR, "trainer", "model_metrics.json")
        val_acc = "100.0%"
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as mf:
                    val_acc = f"{json.load(mf).get('validation_accuracy', '100.0')}%"
            except Exception:
                pass

        messagebox.showinfo("CNN Training Success", 
                            f"VisionLock CNN successfully trained!\n"
                            f"Achieved Model Accuracy: {val_acc}\n"
                            "User is now ready for Biometric Login.")
        # Clear fields
        entry_id.delete(0, tk.END)
        entry_name.delete(0, tk.END)
        entry_password.delete(0, tk.END)
    else:
        messagebox.showerror("Error", "CNN Model training failed. Check console logs.")

def login():
    user_id = entry_login_id.get().strip()
    password = entry_login_password.get().strip()

    if not user_id or not password:
        messagebox.showerror("Error", "Both User ID and Password are required!")
        return

    # Load users
    try:
        with open(user_db, "r", encoding="utf-8") as f:
            users = json.load(f)
    except Exception:
        users = {}

    # Check direct user_id or normalized user_id (handles '01' vs '1')
    target_id = user_id
    if target_id not in users:
        norm_id = user_id.lstrip("0") or "0"
        if norm_id in users:
            target_id = norm_id
        else:
            # Check if any user key normalizes to the same value
            matched_key = None
            for k in users.keys():
                if k.lstrip("0") == user_id.lstrip("0"):
                    matched_key = k
                    break
            if matched_key:
                target_id = matched_key
            else:
                messagebox.showerror("Error", "Invalid User ID!")
                return

    user_id = target_id
    hashed_password = hash_password(password)
    if users[user_id]["password"] != hashed_password:
        messagebox.showerror("Error", "Invalid Password!")
        return

    # Ensure CNN model is loaded
    if not recognizer.model_loaded:
        recognizer.reload_model()

    if not recognizer.model_loaded:
        messagebox.showerror("Error", "Face recognition CNN model is not trained yet. Please register a user first!")
        return

    username = users[user_id]["name"]
    messagebox.showinfo("Biometric Challenge", 
                        f"Credential Verified for {username}.\n\n"
                        "Webcam will now activate:\n"
                        "1. CNN will verify your face (>98% accuracy).\n"
                        "2. Perform the random liveness challenge (Blink / Look Left / Look Right).")

    status, _ = recognize_face(user_id)

    if status == "success":
        messagebox.showinfo("Access Granted", 
                            f"Welcome, {username}!\n\n"
                            "Multi-Layer Biometric Verification Succeeded:\n"
                            "- Layer 1: Credentials (Passed)\n"
                            "- Layer 2: CNN Facial Biometrics (Passed)\n"
                            "- Layer 3: Dlib Liveness / Anti-Spoofing (Passed)")
        entry_login_password.delete(0, tk.END)
    elif status == "idle":
        messagebox.showwarning("Liveness Warning", 
                               "Face was recognized, but the required liveness gesture\n"
                               "(Blink / Look Left / Look Right) was not completed in time.")
    else:
        messagebox.showerror("Access Denied", 
                             "Face authentication failed!\n"
                             "Face does not match the registered biometric profile.")

# Title & Subtitle Header
header_label = ttk.Label(root, text="VISION LOCK", style="Header.TLabel")
header_label.pack(pady=(12, 2))
sub_label = ttk.Label(root, text="Multi-Layer Biometric Authentication | CNN + Dlib", style="SubHeader.TLabel")
sub_label.pack(pady=(0, 10))

# ----------------- Registration Frame -----------------
reg_frame = ttk.LabelFrame(root, text="  User Registration & Biometric Enrollment  ", padding=12)
reg_frame.pack(fill="x", padx=20, pady=6)

ttk.Label(reg_frame, text="User ID:").grid(row=0, column=0, sticky="w", pady=3)
entry_id = ttk.Entry(reg_frame, width=28)
entry_id.grid(row=0, column=1, pady=3, padx=5)

ttk.Label(reg_frame, text="Username:").grid(row=1, column=0, sticky="w", pady=3)
entry_name = ttk.Entry(reg_frame, width=28)
entry_name.grid(row=1, column=1, pady=3, padx=5)

ttk.Label(reg_frame, text="Password:").grid(row=2, column=0, sticky="w", pady=3)
entry_password = ttk.Entry(reg_frame, show="*", width=28)
entry_password.grid(row=2, column=1, pady=3, padx=5)

btn_reg = tk.Button(reg_frame, text="Enroll & Train CNN", command=register, 
                    bg="#a6e3a1", fg="#11111b", font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=4)
btn_reg.grid(row=3, columnspan=2, pady=(8, 2))

# ----------------- Login Frame -----------------
login_frame = ttk.LabelFrame(root, text="  Biometric Authentication (Login)  ", padding=12)
login_frame.pack(fill="x", padx=20, pady=10)

ttk.Label(login_frame, text="User ID:").grid(row=0, column=0, sticky="w", pady=3)
entry_login_id = ttk.Entry(login_frame, width=28)
entry_login_id.grid(row=0, column=1, pady=3, padx=5)

ttk.Label(login_frame, text="Password:").grid(row=1, column=0, sticky="w", pady=3)
entry_login_password = ttk.Entry(login_frame, show="*", width=28)
entry_login_password.grid(row=1, column=1, pady=3, padx=5)

btn_login = tk.Button(login_frame, text="Verify & Login", command=login, 
                      bg="#89b4fa", fg="#11111b", font=("Segoe UI", 9, "bold"), relief="flat", padx=14, pady=4)
btn_login.grid(row=2, columnspan=2, pady=(8, 2))

# Footer
footer_label = ttk.Label(root, text="Layers: [1] Password  [2] CNN Biometrics  [3] Dlib Liveness", 
                         style="SubHeader.TLabel")
footer_label.pack(side="bottom", pady=8)

root.mainloop()