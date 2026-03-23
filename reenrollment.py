import cv2
import numpy as np
import sqlite3
import pickle
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encryption import encrypt_encoding, decrypt_encoding
from deepface import DeepFace

# ─────────────────────────────────────────
# GET ALL REGISTERED USERS
# ─────────────────────────────────────────
def get_all_users():
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, registered_at FROM users')
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        print("❌ No users registered yet!")
        return []

    print("\n👥 Registered Users:")
    print("─" * 40)
    for row in rows:
        print(f"ID: {row[0]} | Name: {row[1]} | Registered: {row[2]}")
    print("─" * 40)
    return rows

# ─────────────────────────────────────────
# RE-ENROLL USER WITH NEW FACE VARIATIONS
# ─────────────────────────────────────────
def reenroll_user(user_id, variation_name="default"):
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"❌ User ID {user_id} not found!")
        return

    name = row[0]
    print(f"\n🔄 Re-enrolling: {name} ({variation_name})")
    print("Look at camera. Capturing 20 photos...")
    print("Press 'q' to quit anytime\n")

    cap = cv2.VideoCapture(0)

    # Create variation folder
    variation_folder = f"data/faces/{name}/{variation_name}"
    os.makedirs(variation_folder, exist_ok=True)

    captured = 0
    embeddings = []

    while captured < 20:
        ret, frame = cap.read()
        if not ret:
            break

        temp_path = f"{variation_folder}/temp.jpg"
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
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

                cv2.putText(frame, f"Re-enrolling: {name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Variation: {variation_name}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Captured: {captured}/20", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imwrite(f"{variation_folder}/{captured}.jpg", frame)

        except Exception:
            cv2.putText(frame, "No face detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Re-enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ─────────────────────────────────────────
    # SAVE NEW ENCODING TO DATABASE
    # ─────────────────────────────────────────
    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0)
        encrypted_encoding = encrypt_encoding(avg_embedding)

        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()

        # Add variation column if not exists
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN variation TEXT DEFAULT "default"')
            conn.commit()
        except:
            pass

        # Insert new encoding for same user
        cursor.execute(
            'INSERT INTO users (name, encoding, variation) VALUES (?, ?, ?)',
            (name, encrypted_encoding, variation_name)
        )
        conn.commit()
        conn.close()

        print(f"\n✅ Re-enrollment complete for {name}!")
        print(f"📸 Variation '{variation_name}' saved successfully!")
    else:
        print("❌ Re-enrollment failed - no face detected!")

# ─────────────────────────────────────────
# ACCURACY MEASUREMENT
# ─────────────────────────────────────────
def measure_accuracy():
    print("\n📊 Measuring System Accuracy...")
    print("─" * 40)

    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, encoding FROM users')
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        print("❌ No users found!")
        return

    users = []
    for row in rows:
        name = row[0]
        encoding = decrypt_encoding(row[1])
        users.append((name, encoding))

    print(f"✅ Loaded {len(users)} face encodings")

    # Calculate distances between all pairs
    print("\n📏 Distance Matrix:")
    print("─" * 40)

    same_distances = []
    diff_distances = []

    for i in range(len(users)):
        for j in range(i+1, len(users)):
            name1, enc1 = users[i]
            name2, enc2 = users[j]
            distance = np.linalg.norm(np.array(enc1) - np.array(enc2))

            if name1 == name2:
                same_distances.append(distance)
                print(f"✅ SAME person ({name1}): distance = {distance:.2f}")
            else:
                diff_distances.append(distance)
                print(f"❌ DIFF person ({name1} vs {name2}): distance = {distance:.2f}")

    print("\n📊 Accuracy Statistics:")
    print("─" * 40)

    if same_distances:
        print(f"Average distance (same person): {np.mean(same_distances):.2f}")
        print(f"Max distance (same person): {np.max(same_distances):.2f}")

    if diff_distances:
        print(f"Average distance (different people): {np.mean(diff_distances):.2f}")
        print(f"Min distance (different people): {np.min(diff_distances):.2f}")

    # Suggest optimal threshold
    if same_distances and diff_distances:
        suggested_threshold = (np.max(same_distances) + np.min(diff_distances)) / 2
        print(f"\n🎯 Suggested optimal threshold: {suggested_threshold:.2f}")
        print("Use this value in recognize_face.py for best accuracy!")

# ─────────────────────────────────────────
# REENROLLMENT REMINDER SYSTEM
# ─────────────────────────────────────────
def check_reenrollment_reminder():
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, MAX(registered_at) as last_enrolled
        FROM users
        GROUP BY name
    ''')
    rows = cursor.fetchall()
    conn.close()

    print("\n⏰ Re-enrollment Status:")
    print("─" * 40)

    for row in rows:
        name = row[0]
        last_enrolled = row[1]
        print(f"👤 {name} | Last enrolled: {last_enrolled}")
        print(f"   💡 Tip: Re-enroll monthly for best accuracy!")

# ─────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────
def main():
    while True:
        print("\n" + "="*40)
        print("🔄 RE-ENROLLMENT SYSTEM")
        print("="*40)
        print("1. View all registered users")
        print("2. Re-enroll with new variation")
        print("3. Measure system accuracy")
        print("4. Check re-enrollment reminders")
        print("5. Exit")
        print("="*40)

        choice = input("Enter choice (1-5): ")

        if choice == "1":
            get_all_users()

        elif choice == "2":
            users = get_all_users()
            if users:
                user_id = input("\nEnter user ID to re-enroll: ")
                print("\nVariation types:")
                print("1. glasses")
                print("2. beard")
                print("3. mask")
                print("4. different_angle")
                print("5. custom")
                var_choice = input("Enter variation (or custom name): ")
                variations = {
                    "1": "glasses",
                    "2": "beard",
                    "3": "mask",
                    "4": "different_angle"
                }
                variation = variations.get(var_choice, var_choice)
                reenroll_user(int(user_id), variation)

        elif choice == "3":
            measure_accuracy()

        elif choice == "4":
            check_reenrollment_reminder()

        elif choice == "5":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
