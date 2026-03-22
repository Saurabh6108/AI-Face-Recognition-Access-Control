# 🔐 AI-Based Face Recognition Access Control System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![DeepFace](https://img.shields.io/badge/DeepFace-FaceNet-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Flask](https://img.shields.io/badge/Flask-API-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A research-level Final Year Project that protects confidential folders and data using real-time AI-powered Face Recognition, Liveness Detection, and Biometric Encryption.

---

## 📌 Project Overview

This system provides **biometric-based access control** for protecting confidential folders and sensitive data on a computer. Only authorized users can unlock protected folders — verified through real-time face recognition combined with liveness detection.

Inspired by how modern smartphones like **Apple Face ID** and **Android Face Unlock** work, this project brings the same technology to desktop computers with added security enhancements.

---

## 🎯 Key Features

- 🎥 **Real-Time Face Detection** — Live webcam-based face detection using OpenCV
- 👤 **Face Registration** — Capture and store face encodings using FaceNet (DeepFace)
- 👁️ **Liveness Detection** — Prevents spoofing attacks using blink detection and head movement challenges
- 🌙 **Image Preprocessing** — Works in low light, bright, and varied lighting conditions
- 🔒 **Biometric Encryption** — Face encodings encrypted using Fernet (AES-128) cryptography
- 📁 **Folder Encryption** — Protected folders are fully encrypted and hidden when locked
- 🔄 **Re-enrollment System** — Handles face changes (glasses, beard, mask, aging)
- 📊 **Accuracy Measurement** — FAR, FRR, and threshold tuning
- 🌍 **Bias Handling** — Tested on diverse datasets for fair recognition
- 🖥️ **Modern GUI** — Beautiful dark-themed interface built with CustomTkinter
- 🌐 **REST API** — Flask-based API for future mobile app integration
- 📝 **Access Logs** — Complete audit trail of all access attempts
- 🚨 **Security Alerts** — Alerts after multiple failed attempts

---

## 🛡️ Security Drawbacks Addressed

| # | Drawback | Solution |
|---|---|---|
| 1 | Spoofing Attack | Liveness Detection (Blink + Head Movement) |
| 2 | Low Light Conditions | CLAHE + Auto Brightness Preprocessing |
| 3 | Face Changes Over Time | Re-enrollment + Variation Training |
| 4 | Privacy Concerns | Fernet Encryption of Biometric Data |
| 5 | High False Accept/Reject Rate | DeepFace FaceNet + Threshold Tuning |
| 6 | Skin Tone Bias | Diverse Dataset Training + Fairness Metrics |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Face Recognition | DeepFace (FaceNet Model) |
| Face Detection | OpenCV + Haar Cascade |
| Liveness Detection | MediaPipe Face Mesh |
| Encryption | Python Cryptography (Fernet) |
| GUI | CustomTkinter |
| Database | SQLite3 |
| API | Flask + Flask-CORS |
| Image Processing | OpenCV + NumPy + Pillow |

---

## 📁 Project Structure

```
FaceRecognitionProject/
│
├── data/
│   ├── faces/           # Registered face images
│   └── database.db      # Encrypted face encodings database
│
├── modules/
│   ├── preprocessing.py     # Image preprocessing module
│   ├── liveness.py          # Liveness detection module
│   ├── recognition.py       # Face recognition engine
│   ├── encryption.py        # Biometric data encryption
│   ├── reenrollment.py      # Re-enrollment system
│   └── access_control.py    # Access control logic
│
├── gui/
│   └── main_window.py       # GUI application
│
├── logs/
│   └── access_logs.txt      # Access attempt logs
│
├── register_face.py         # Face registration script
├── recognize_face.py        # Face recognition script
├── api.py                   # Flask REST API
├── main.py                  # Entry point
└── requirements.txt         # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Webcam

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/AI-Face-Recognition-Access-Control.git
cd AI-Face-Recognition-Access-Control
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies:**
```bash
pip install opencv-python deepface tf-keras numpy pillow cryptography pyqt5 customtkinter flask flask-cors mediapipe
```

**4. Register your face:**
```bash
python register_face.py
```

**5. Run the GUI:**
```bash
python gui\main_window.py
```

**6. Or run Access Control directly:**
```bash
python modules\access_control.py
```

---

## 🔐 How It Works

```
User Opens App
      ↓
Liveness Challenge
(Blink / Turn Head)
      ↓
Face Detection
      ↓
Image Preprocessing
(Fix Lighting)
      ↓
FaceNet Embedding
      ↓
Compare with Encrypted Database
      ↓
Match Found?
   ↓         ↓
  YES         NO
   ↓         ↓
Decrypt    Deny Access
Folder     Log Attempt
   ↓
Access Granted!
```

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|---|---|---|
| Recognition Accuracy | > 95% | ✅ |
| Liveness Detection | > 98% | ✅ |
| False Acceptance Rate | < 1% | ✅ |
| False Rejection Rate | < 5% | ✅ |
| Processing Time | < 2 sec | ✅ |

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /ping | Test connection |
| POST | /register | Register new face |
| POST | /recognize | Recognize face |
| GET | /users | Get all users |
| GET | /logs | Get access logs |
| DELETE | /users/{id} | Delete user |

---

## 📸 Screenshots

> GUI Home Screen, Registration Screen, Login Screen, Admin Dashboard

---

## 🔮 Future Scope

- 📱 Mobile app using Flutter + TensorFlow Lite
- ☁️ Cloud-based identity management
- 🔑 Multi-factor authentication (Face + OTP)
- 📷 3D face recognition using depth cameras
- 🌐 Network-level access control for enterprises

---

## 👨‍💻 Author

**Saurabh**
- Final Year Computer Science Student
- Project: AI-Based Face Recognition Access Control System

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project helpful, please give it a star!
