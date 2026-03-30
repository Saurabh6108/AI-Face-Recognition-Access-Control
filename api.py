from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import sqlite3
import os
import sys
import base64
import threading
sys.path.append('modules')
from encryption import encrypt_encoding, decrypt_encoding
from preprocessing import preprocess_image
from liveness import LivenessDetector
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

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

        return users
    except Exception as e:
        print(f"Error loading users: {e}")
        return []

# ─────────────────────────────────────────
# DECODE BASE64 IMAGE FROM APP
# ─────────────────────────────────────────
def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# ─────────────────────────────────────────
# API ROUTE 1: TEST CONNECTION
# ─────────────────────────────────────────
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "success",
        "message": "Face Recognition API is running!",
        "version": "1.0"
    })

# ─────────────────────────────────────────
# API ROUTE 2: REGISTER FACE
# ─────────────────────────────────────────
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        name = data.get('name')
        image_base64 = data.get('image')

        if not name or not image_base64:
            return jsonify({
                "status": "error",
                "message": "Name and image are required!"
            }), 400

        # Decode image
        image = decode_image(image_base64)

        # Save temp image
        os.makedirs('data/temp', exist_ok=True)
        temp_path = 'data/temp/register_temp.jpg'
        cv2.imwrite(temp_path, image)

        # Get face embedding
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )

        if not result:
            return jsonify({
                "status": "error",
                "message": "No face detected in image!"
            }), 400

        embedding = np.array(result[0]["embedding"])

        # Encrypt and save to database
        encrypted_encoding = encrypt_encoding(embedding)

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
        cursor.execute(
            'INSERT INTO users (name, encoding) VALUES (?, ?)',
            (name, encrypted_encoding)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": f"{name} registered successfully!",
            "name": name
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ─────────────────────────────────────────
# API ROUTE 3: RECOGNIZE FACE
# ─────────────────────────────────────────
@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        image_base64 = data.get('image')

        if not image_base64:
            return jsonify({
                "status": "error",
                "message": "Image is required!"
            }), 400

        # Decode image
        image = decode_image(image_base64)

        # Preprocess image
        processed = preprocess_image(image)

        # Save temp image
        os.makedirs('data/temp', exist_ok=True)
        temp_path = 'data/temp/recognize_temp.jpg'
        cv2.imwrite(temp_path, processed)

        # Get face embedding
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )

        if not result:
            return jsonify({
                "status": "error",
                "message": "No face detected!"
            }), 400

        current_embedding = np.array(result[0]["embedding"])

        # Compare with all users
        users = load_users()
        best_match = None
        best_distance = float('inf')
        threshold = 10.0

        for name, stored_embedding in users:
            distance = np.linalg.norm(current_embedding - stored_embedding)
            if distance < threshold and distance < best_distance:
                best_match = name
                best_distance = distance

        # Log access attempt
        os.makedirs('logs', exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if best_match:
            log_entry = f"[{timestamp}] User: {best_match} | Status: GRANTED | Distance: {best_distance:.2f}\n"
            with open('logs/access_logs.txt', 'a', encoding='utf-8') as f:
                f.write(log_entry)

            return jsonify({
                "status": "success",
                "recognized": True,
                "name": best_match,
                "distance": round(best_distance, 2),
                "message": f"Welcome {best_match}!"
            })
        else:
            log_entry = f"[{timestamp}] User: Unknown | Status: DENIED\n"
            with open('logs/access_logs.txt', 'a', encoding='utf-8') as f:
                f.write(log_entry)

            return jsonify({
                "status": "success",
                "recognized": False,
                "name": None,
                "message": "Face not recognized!"
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ─────────────────────────────────────────
# API ROUTE 4: GET ALL USERS
# ─────────────────────────────────────────
@app.route('/users', methods=['GET'])
def get_users():
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, registered_at FROM users')
        rows = cursor.fetchall()
        conn.close()

        users = []
        for row in rows:
            users.append({
                "id": row[0],
                "name": row[1],
                "registered_at": row[2]
            })

        return jsonify({
            "status": "success",
            "users": users,
            "count": len(users)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ─────────────────────────────────────────
# API ROUTE 5: GET ACCESS LOGS
# ─────────────────────────────────────────
@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        log_path = 'logs/access_logs.txt'
        if not os.path.exists(log_path):
            return jsonify({
                "status": "success",
                "logs": [],
                "message": "No logs yet"
            })

        with open(log_path, 'r', encoding='utf-8') as f:
            logs = f.readlines()

        logs = [log.strip() for log in logs if log.strip()]
        logs.reverse()

        return jsonify({
            "status": "success",
            "logs": logs[:50],
            "count": len(logs)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ─────────────────────────────────────────
# API ROUTE 6: DELETE USER
# ─────────────────────────────────────────
@app.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": f"User {user_id} deleted successfully!"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ─────────────────────────────────────────
# RUN API SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Face Recognition API Server...")
    print("─" * 40)
    print("📡 API running at: http://localhost:5000")
    print("─" * 40)
    print("Available endpoints:")
    print("  GET  /ping        - Test connection")
    print("  POST /register    - Register face")
    print("  POST /recognize   - Recognize face")
    print("  GET  /users       - Get all users")
    print("  GET  /logs        - Get access logs")
    print("  DELETE /users/id  - Delete user")
    print("─" * 40)
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context='adhoc')
