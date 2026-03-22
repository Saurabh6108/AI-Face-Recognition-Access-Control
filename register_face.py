import cv2
import numpy as np
import sqlite3
import os
import sys
sys.path.append('modules')
from encryption import encrypt_encoding
from deepface import DeepFace

# ─────────────────────────────────────────
# SETUP DATABASE
# ─────────────────────────────────────────
def setup_database():
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database setup complete!")

# ─────────────────────────────────────────
# CAPTURE AND REGISTER FACE
# ─────────────────────────────────────────
def register_face(name):
    print(f"\n📸 Registering face for: {name}")
    print("Look at the camera. Capturing 30 photos...")
    print("Press 'q' to quit anytime\n")

    cap = cv2.VideoCapture(0)

    user_folder = f"data/faces/{name}"
    os.makedirs(user_folder, exist_ok=True)

    captured = 0
    embeddings = []

    while captured < 30:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        frame = cv2.resize(frame, (640, 480))
        temp_path = f"{user_folder}/temp.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            result = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=True
            )

            if result:
                embedding = result[0]["embedding"]
                embeddings.append(embedding)
                captured += 1

                facial_area = result[0]["facial_area"]
                x, y = facial_area["x"], facial_area["y"]
                w, h = facial_area["w"], facial_area["h"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.putText(frame, f"Captured: {captured}/30", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Registering: {name}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.imwrite(f"{user_folder}/{captured}.jpg", frame)

        except Exception:
            cv2.putText(frame, "No face detected - move closer!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ─────────────────────────────────────────
    # SAVE ENCRYPTED ENCODING TO DATABASE
    # ─────────────────────────────────────────
    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0)

        # Encrypt the encoding 🔒
        encrypted_encoding = encrypt_encoding(avg_embedding)

        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, encoding) VALUES (?, ?)',
            (name, encrypted_encoding)
        )
        conn.commit()
        conn.close()

        print(f"\n✅ Face registered successfully for {name}!")
        print(f"📸 Captured {len(embeddings)} photos")
        print(f"🔒 Encrypted encoding saved to database!")
    else:
        print("❌ Registration failed - no face detected!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    setup_database()
    name = input("\nEnter your name: ")
    register_face(name)