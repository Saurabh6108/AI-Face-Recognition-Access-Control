import cv2
import numpy as np
import sqlite3
import pickle
import os
import sys
import time
import zipfile
import shutil
from datetime import datetime
from cryptography.fernet import Fernet

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from encryption import encrypt_encoding, decrypt_encoding, load_key
from preprocessing import preprocess_image
from liveness import LivenessDetector
from deepface import DeepFace


# ─────────────────────────────────────────
# ACCESS LOGGER
# ─────────────────────────────────────────
def log_access(name, status, reason=""):
    log_path = "logs/access_logs.txt"
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] User: {name} | Status: {status} | Reason: {reason}\n"
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)
    print(f"📝 Logged: {log_entry.strip()}")


# ─────────────────────────────────────────
# LOAD USERS FROM DATABASE
# ─────────────────────────────────────────
def load_users():
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name, encoding FROM users')
        rows = cursor.fetchall()
        conn.close()

        users = []
        for row in rows:
            name = row[0]
            encoding = decrypt_encoding(row[1])
            users.append((name, np.array(encoding)))

        print(f"✅ Loaded {len(users)} user(s) from database")
        return users
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        return []


# ─────────────────────────────────────────
# ENCRYPT FOLDER
# ─────────────────────────────────────────
def encrypt_folder(folder_path):
    key = load_key()
    fernet = Fernet(key)

    # Zip entire folder
    zip_path = folder_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path,
                           os.path.relpath(file_path, folder_path))

    # Encrypt the zip
    with open(zip_path, 'rb') as f:
        data = f.read()

    encrypted_data = fernet.encrypt(data)

    # Save encrypted file
    encrypted_path = folder_path + ".locked"
    with open(encrypted_path, 'wb') as f:
        f.write(encrypted_data)

    # Delete original folder and zip
    shutil.rmtree(folder_path)
    os.remove(zip_path)

    print(f"🔒 Folder encrypted: {encrypted_path}")
    return encrypted_path


# ─────────────────────────────────────────
# DECRYPT FOLDER
# ─────────────────────────────────────────
def decrypt_folder(encrypted_path):
    key = load_key()
    fernet = Fernet(key)

    # Read encrypted file
    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()

    # Decrypt
    decrypted_data = fernet.decrypt(encrypted_data)

    # Save as zip
    zip_path = encrypted_path.replace(".locked", ".zip")
    with open(zip_path, 'wb') as f:
        f.write(decrypted_data)

    # Extract zip to original folder
    folder_path = encrypted_path.replace(".locked", "")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(folder_path)

    # Delete encrypted file and zip
    os.remove(encrypted_path)
    os.remove(zip_path)

    print(f"🔓 Folder decrypted: {folder_path}")
    return folder_path


# ─────────────────────────────────────────
# LOCK FOLDER
# ─────────────────────────────────────────
def lock_folder(folder_path):
    encrypted_path = folder_path + ".locked"

    if os.path.exists(folder_path):
        encrypt_folder(folder_path)
        print(f"🔒 Folder locked and encrypted!")
        return encrypted_path
    elif os.path.exists(encrypted_path):
        print(f"🔒 Folder already locked!")
        return encrypted_path
    else:
        print(f"❌ Folder not found: {folder_path}")
        return None


# ─────────────────────────────────────────
# UNLOCK FOLDER
# ─────────────────────────────────────────
def unlock_folder(locked_path):
    encrypted_path = locked_path
    original_path = locked_path.replace(".locked", "")

    if os.path.exists(encrypted_path):
        decrypt_folder(encrypted_path)
        print(f"🔓 Folder unlocked and decrypted!")
        return original_path
    else:
        print(f"❌ Encrypted folder not found!")
        return None


# ─────────────────────────────────────────
# ACCESS CONTROL SYSTEM CLASS
# ─────────────────────────────────────────
class AccessControlSystem:

    def __init__(self, protected_folder="SecureFolder"):
        self.protected_folder = protected_folder
        self.locked_folder = protected_folder + ".locked"
        self.users = load_users()
        self.threshold = 10.0
        self.max_failed_attempts = 3
        self.failed_attempts = 0
        self.liveness_detector = LivenessDetector()
        self.liveness_confirmed = False
        self.recognized_user = None

        # Lock the folder on startup
        if os.path.exists(self.protected_folder):
            lock_folder(self.protected_folder)
            print(f"🔒 Folder locked!")
        elif os.path.exists(self.locked_folder):
            print(f"🔒 Folder is already locked!")
        else:
            print(f"❌ Folder not found: {self.protected_folder}")

    def check_liveness(self, frame):
        frame, is_live, status = self.liveness_detector.check_liveness(frame)
        if is_live:
            self.liveness_confirmed = True
            print("✅ Liveness confirmed!")
        return frame, is_live

    def recognize_user(self, frame):
        processed_frame = preprocess_image(frame.copy())
        cv2.imwrite("data/temp_auth.jpg", processed_frame)

        try:
            result = DeepFace.represent(
                img_path="data/temp_auth.jpg",
                model_name="Facenet",
                enforce_detection=True
            )

            if result:
                current_embedding = np.array(result[0]["embedding"])
                facial_area = result[0]["facial_area"]
                x, y = facial_area["x"], facial_area["y"]
                w, h = facial_area["w"], facial_area["h"]

                best_match = None
                best_distance = float('inf')

                for name, stored_embedding in self.users:
                    distance = np.linalg.norm(
                        current_embedding - stored_embedding
                    )
                    if distance < self.threshold and distance < best_distance:
                        best_match = name
                        best_distance = distance

                if best_match:
                    self.recognized_user = best_match
                    return True, best_match, best_distance, (x, y, w, h)
                else:
                    return False, "Unknown", best_distance, (x, y, w, h)

        except Exception as e:
            pass

        return False, None, None, None

    def grant_access(self, name):
        print(f"\n✅ ACCESS GRANTED to {name}!")
        log_access(name, "GRANTED", "Face recognized + Liveness confirmed")

        if os.path.exists(self.locked_folder):
            unlock_folder(self.locked_folder)
            print(f"🔓 Folder unlocked for {name}!")

        self.failed_attempts = 0
        return True

    def deny_access(self, reason="Unknown person"):
        self.failed_attempts += 1
        print(f"\n❌ ACCESS DENIED! Reason: {reason}")
        print(f"⚠️ Failed attempts: {self.failed_attempts}/{self.max_failed_attempts}")
        log_access("Unknown", "DENIED", reason)

        if self.failed_attempts >= self.max_failed_attempts:
            print("🚨 ALERT! Maximum failed attempts reached!")
            log_access("Unknown", "ALERT", "Maximum failed attempts reached!")

        return False

    def run(self):
        print("\n🔐 Access Control System Started")
        print("─" * 40)
        print("Phase 1: Complete liveness challenge")
        print("Phase 2: Face will be recognized")
        print("Press 'q' to quit\n")

        cap = cv2.VideoCapture(0)
        access_granted = False
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()
            frame_count += 1

            # PHASE 1: LIVENESS CHECK
            if not self.liveness_confirmed:
                display_frame, is_live = self.check_liveness(display_frame)
                cv2.putText(display_frame, "PHASE 1: Prove you are REAL!",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # PHASE 2: FACE RECOGNITION
            elif not access_granted:
                if frame_count % 30 == 0:
                    recognized, name, distance, bbox = self.recognize_user(frame)

                    if recognized and name:
                        access_granted = self.grant_access(name)
                    elif name == "Unknown":
                        self.deny_access("Face not recognized")

                cv2.putText(display_frame, "PHASE 2: Recognizing face...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, "Please look at camera",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ACCESS GRANTED SCREEN
            else:
                cv2.putText(display_frame,
                            f"WELCOME {self.recognized_user}!",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(display_frame,
                            "Folder is now UNLOCKED!",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame,
                            "Press 'l' to lock | 'q' to quit",
                            (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show failed attempts
            if self.failed_attempts > 0:
                cv2.putText(display_frame,
                            f"Failed: {self.failed_attempts}/{self.max_failed_attempts}",
                            (10, display_frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Recognition Access Control", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if os.path.exists(self.protected_folder):
                    lock_folder(self.protected_folder)
                break
            elif key == ord('l') and access_granted:
                if os.path.exists(self.protected_folder):
                    lock_folder(self.protected_folder)
                    access_granted = False
                    self.liveness_confirmed = False
                    self.liveness_detector = LivenessDetector()
                    print("🔒 Folder locked again!")

        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Access Control System closed!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    system = AccessControlSystem(protected_folder="D:\\Baby")
    system.run()