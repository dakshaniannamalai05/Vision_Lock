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

def capture_face_images(user_id, save_path=None, num_images=40):
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "dataset")

    if os.path.exists(CASCADE_PATH):
        face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    else:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Cannot open webcam.")
        return False

    user_folder = os.path.join(save_path, f"user_{user_id}")

    # Check if user already exists
    if os.path.exists(user_folder) and len(os.listdir(user_folder)) > 0:
        print(f"[INFO] User {user_id} already has dataset images. Updating dataset...")

    os.makedirs(user_folder, exist_ok=True)

    count = 0
    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

        for (x, y, fw, fh) in faces:
            count += 1
            file_name = os.path.join(user_folder, f"{count}.jpg")
            # Save BGR face crop for CNN representation
            cv2.imwrite(file_name, frame[y:y+fh, x:x+fw])
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample: {count}/{num_images}", (x, max(20, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"VISION LOCK - Face Capture: User {user_id}", (15, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Capturing Faces - Press Q to Quit", frame)

        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"[SUCCESS] {count} images saved for User ID: {user_id}")
        return True
    else:
        print("[ERROR] No images captured. Try again.")
        return False

if __name__ == "__main__":
    uid = input("Enter user ID to capture: ").strip()
    if uid:
        capture_face_images(uid)