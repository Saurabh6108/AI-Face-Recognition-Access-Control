import cv2
import numpy as np
import sqlite3
import pickle
from deepface import DeepFace

# ─────────────────────────────────────────
# STEP 1: Load all users from database
# ─────────────────────────────────────────
def load_users():
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, encoding FROM users')
    rows = cursor.fetchall()
    conn.close()

    users = []
    for row in rows:
        name = row[0]
        encoding = pickle.loads(row[1])
        users.append((name, encoding))

    print(f"✅ Loaded {len(users)} user(s) from database")
    return users

# ─────────────────────────────────────────
# STEP 2: Compare two face embeddings
# ─────────────────────────────────────────
def compare_faces(embedding1, embedding2, threshold=10):
    # Calculate distance between two embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    print(f"Distance: {distance:.2f}")
    return distance < threshold, distance

# ─────────────────────────────────────────
# STEP 3: Real time face recognition
# ─────────────────────────────────────────
def recognize_face():
    print("\n🎥 Starting Face Recognition...")
    print("Press 'q' to quit\n")

    users = load_users()

    if len(users) == 0:
        print("❌ No users registered! Please run register_face.py first.")
        return

    cap = cv2.VideoCapture(0)
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 10th frame for performance
        frame_count += 1
        if frame_count % 10 != 0:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Save temp image
        cv2.imwrite("data/temp.jpg", frame)

        try:
            # Get embedding of current face
            result = DeepFace.represent(
                img_path="data/temp.jpg",
                model_name="Facenet",
                enforce_detection=True
            )

            if result:
                current_embedding = np.array(result[0]["embedding"])
                facial_area = result[0]["facial_area"]
                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]

                # Compare with all registered users
                best_match = "Unknown"
                best_distance = float('inf')

                for name, stored_embedding in users:
                    match, distance = compare_faces(
                        current_embedding, stored_embedding
                    )
                    if match and distance < best_distance:
                        best_match = name
                        best_distance = distance

                # Draw result
                if best_match != "Unknown":
                    # Green box - recognized
                    color = (0, 255, 0)
                    label = f"✅ {best_match} ({best_distance:.2f})"
                else:
                    # Red box - unknown
                    color = (0, 0, 255)
                    label = "❌ Unknown Person"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    recognize_face()