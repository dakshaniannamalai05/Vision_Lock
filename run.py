import os
import sys
import subprocess

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

root_dir = os.path.dirname(os.path.abspath(__file__))
faceauth_dir = os.path.join(root_dir, "FaceAuth")
main_script = os.path.join(faceauth_dir, "main.py")

if os.path.exists(main_script):
    print(f"[INFO] Starting Vision Lock GUI from {main_script}...")
    subprocess.run([sys.executable, main_script], cwd=faceauth_dir)
else:
    print(f"[ERROR] Could not find main.py at {main_script}")
