import cv2
import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

if os.path.exists(CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def register_face(user_id, num_samples=40):
    """
    Captures facial biometric training samples for the specified user_id.
    Saves samples in dataset/user_<user_id>/ with real-time UI progress.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam for registration.")
        return False

    count = 0
    save_path = os.path.join(BASE_DIR, "dataset", f"user_{user_id}")
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] Registering Face for User ID: {user_id}. Capturing {num_samples} samples...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture webcam image.")
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        for (x, y, fw, fh) in faces:
            count += 1
            # Save BGR face crop for CNN representation
            face_crop = frame[y:y+fh, x:x+fw]
            img_file = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_file, face_crop)

            # Draw bounding box and progress
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {count}/{num_samples}", (x, max(20, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Header overlay
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        progress_pct = int((count / num_samples) * 100)
        cv2.putText(frame, f"VISION LOCK - Registering User ID: {user_id} ({progress_pct}%)", 
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Register Face - Look at Camera (Press Q to Cancel)", frame)

        if cv2.waitKey(80) & 0xFF == ord('q'):
            print("[INFO] Face registration cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= num_samples:
        print(f"[SUCCESS] {count} images saved for User ID: {user_id}")
        return True
    else:
        print(f"[WARNING] Only {count}/{num_samples} images captured.")
        return count > 0


if __name__ == "__main__":
    uid = input("Enter User ID to register: ")
    if uid.strip():
        register_face(uid.strip())