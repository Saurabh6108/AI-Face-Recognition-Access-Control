import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import sqlite3
import os
import sys
import threading
from PIL import Image, ImageTk
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modules'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from encryption import encrypt_encoding, decrypt_encoding
from preprocessing import preprocess_image
from liveness import LivenessDetector
from deepface import DeepFace

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────
class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window settings
        self.title("🔐 Face Recognition Access Control System")
        self.geometry("1100x700")
        self.resizable(False, False)

        # Camera
        self.cap = None
        self.camera_running = False
        self.current_frame = None

        # State
        self.current_screen = None
        self.liveness_detector = None
        self.liveness_confirmed = False
        self.users = []
        self.failed_attempts = 0
        self.max_failed = 3

        # Load users
        self.load_users()

        # Show home screen
        self.show_home_screen()

    # ─────────────────────────────────────────
    # LOAD USERS FROM DATABASE
    # ─────────────────────────────────────────
    def load_users(self):
        try:
            conn = sqlite3.connect('data/database.db')
            cursor = conn.cursor()
            cursor.execute('SELECT name, encoding FROM users')
            rows = cursor.fetchall()
            conn.close()

            self.users = []
            for row in rows:
                name = row[0]
                encoding = decrypt_encoding(row[1])
                self.users.append((name, np.array(encoding)))

            print(f"✅ Loaded {len(self.users)} user(s)")
        except Exception as e:
            print(f"⚠️ Could not load users: {e}")

    # ─────────────────────────────────────────
    # CLEAR SCREEN
    # ─────────────────────────────────────────
    def clear_screen(self):
        self.stop_camera()
        for widget in self.winfo_children():
            widget.destroy()

    # ─────────────────────────────────────────
    # HOME SCREEN
    # ─────────────────────────────────────────
    def show_home_screen(self):
        self.clear_screen()
        self.current_screen = "home"

        # Background frame
        main_frame = ctk.CTkFrame(self, fg_color="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # Left panel
        left_panel = ctk.CTkFrame(main_frame, fg_color="#16213e", width=400)
        left_panel.pack(side="left", fill="y", padx=20, pady=20)
        left_panel.pack_propagate(False)

        # Logo and title
        ctk.CTkLabel(left_panel, text="🔐", font=("Arial", 80)).pack(pady=(60, 10))
        ctk.CTkLabel(left_panel, text="Face Recognition",
                    font=("Arial", 24, "bold"),
                    text_color="#00d4ff").pack()
        ctk.CTkLabel(left_panel, text="Access Control System",
                    font=("Arial", 18),
                    text_color="#ffffff").pack(pady=(0, 10))
        ctk.CTkLabel(left_panel, text="Final Year Project",
                    font=("Arial", 14),
                    text_color="#888888").pack()

        # Divider
        ctk.CTkFrame(left_panel, height=2, fg_color="#00d4ff").pack(
            fill="x", padx=30, pady=30)

        # Stats
        conn = sqlite3.connect('data/database.db') if os.path.exists('data/database.db') else None
        user_count = 0
        if conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(DISTINCT name) FROM users')
            user_count = cursor.fetchone()[0]
            conn.close()

        ctk.CTkLabel(left_panel, text=f"👥 Registered Users: {user_count}",
                    font=("Arial", 14),
                    text_color="#aaaaaa").pack(pady=5)
        ctk.CTkLabel(left_panel, text="🛡️ Security: ACTIVE",
                    font=("Arial", 14),
                    text_color="#00ff88").pack(pady=5)
        ctk.CTkLabel(left_panel,
                    text=f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    font=("Arial", 14),
                    text_color="#aaaaaa").pack(pady=5)

        # Right panel
        right_panel = ctk.CTkFrame(main_frame, fg_color="#1a1a2e")
        right_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(right_panel, text="Welcome!",
                    font=("Arial", 32, "bold"),
                    text_color="#ffffff").pack(pady=(80, 10))
        ctk.CTkLabel(right_panel,
                    text="Please select an option to continue",
                    font=("Arial", 16),
                    text_color="#888888").pack(pady=(0, 50))

        # Buttons
        ctk.CTkButton(right_panel,
                     text="🔐  Login with Face",
                     font=("Arial", 18, "bold"),
                     height=60, width=350,
                     fg_color="#00d4ff",
                     text_color="#000000",
                     hover_color="#00a8cc",
                     command=self.show_login_screen).pack(pady=10)

        ctk.CTkButton(right_panel,
                     text="📝  Register New Face",
                     font=("Arial", 18, "bold"),
                     height=60, width=350,
                     fg_color="#00ff88",
                     text_color="#000000",
                     hover_color="#00cc6a",
                     command=self.show_register_screen).pack(pady=10)

        ctk.CTkButton(right_panel,
                     text="📊  Admin Dashboard",
                     font=("Arial", 18, "bold"),
                     height=60, width=350,
                     fg_color="#7c3aed",
                     text_color="#ffffff",
                     hover_color="#6d28d9",
                     command=self.show_admin_screen).pack(pady=10)

        ctk.CTkButton(right_panel,
                     text="❌  Exit",
                     font=("Arial", 16),
                     height=45, width=350,
                     fg_color="#333333",
                     text_color="#ffffff",
                     hover_color="#555555",
                     command=self.quit_app).pack(pady=10)

    # ─────────────────────────────────────────
    # REGISTRATION SCREEN
    # ─────────────────────────────────────────
    def show_register_screen(self):
        self.clear_screen()
        self.current_screen = "register"

        main_frame = ctk.CTkFrame(self, fg_color="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(main_frame, fg_color="#16213e", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkButton(header, text="← Back",
                     font=("Arial", 14),
                     width=100, height=40,
                     fg_color="#333333",
                     command=self.show_home_screen).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(header, text="📝 Register New Face",
                    font=("Arial", 22, "bold"),
                    text_color="#00ff88").pack(side="left", padx=20, pady=15)

        # Content
        content = ctk.CTkFrame(main_frame, fg_color="#1a1a2e")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Camera frame
        self.camera_label = ctk.CTkLabel(content, text="",
                                         width=500, height=400)
        self.camera_label.pack(side="left", padx=20)

        # Right panel
        right = ctk.CTkFrame(content, fg_color="#16213e", width=350)
        right.pack(side="right", fill="y", padx=20)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Enter Your Name",
                    font=("Arial", 18, "bold"),
                    text_color="#ffffff").pack(pady=(40, 10))

        self.name_entry = ctk.CTkEntry(right,
                                       placeholder_text="Your full name",
                                       font=("Arial", 16),
                                       height=45, width=280)
        self.name_entry.pack(pady=10)

        self.reg_status = ctk.CTkLabel(right, text="Enter name and click Start",
                                       font=("Arial", 14),
                                       text_color="#888888",
                                       wraplength=280)
        self.reg_status.pack(pady=20)

        self.progress_label = ctk.CTkLabel(right, text="Progress: 0/30",
                                           font=("Arial", 16, "bold"),
                                           text_color="#00d4ff")
        self.progress_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(right, width=280)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        ctk.CTkButton(right,
                     text="▶  Start Registration",
                     font=("Arial", 16, "bold"),
                     height=50, width=280,
                     fg_color="#00ff88",
                     text_color="#000000",
                     command=self.start_registration).pack(pady=20)

        # Start camera
        self.start_camera()

    def start_registration(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter your name first!")
            return

        self.reg_name = name
        self.reg_captured = 0
        self.reg_embeddings = []
        self.reg_running = True
        self.reg_status.configure(
            text=f"Registering {name}...\nLook at the camera!",
            text_color="#00ff88"
        )

        user_folder = f"data/faces/{name}"
        os.makedirs(user_folder, exist_ok=True)
        self.reg_folder = user_folder

        threading.Thread(target=self.registration_thread, daemon=True).start()

    def registration_thread(self):
        while self.reg_running and self.reg_captured < 30:
            if self.current_frame is None:
                continue

            frame = self.current_frame.copy()
            temp_path = f"{self.reg_folder}/temp.jpg"
            cv2.imwrite(temp_path, frame)

            try:
                result = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True
                )

                if result:
                    embedding = result[0]["embedding"]
                    self.reg_embeddings.append(embedding)
                    self.reg_captured += 1

                    # Update UI
                    self.progress_label.configure(
                        text=f"Progress: {self.reg_captured}/30"
                    )
                    self.progress_bar.set(self.reg_captured / 30)

            except Exception:
                pass

        if self.reg_captured >= 30:
            self.save_registration()

    def save_registration(self):
        avg_embedding = np.mean(self.reg_embeddings, axis=0)
        encrypted_encoding = encrypt_encoding(avg_embedding)

        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, encoding) VALUES (?, ?)',
            (self.reg_name, encrypted_encoding)
        )
        conn.commit()
        conn.close()

        self.load_users()
        self.reg_status.configure(
            text=f"✅ {self.reg_name} registered successfully!",
            text_color="#00ff88"
        )
        self.progress_label.configure(text="Registration Complete! ✅")
        messagebox.showinfo("Success",
                           f"✅ {self.reg_name} registered successfully!")

    # ─────────────────────────────────────────
    # LOGIN SCREEN
    # ─────────────────────────────────────────
    def show_login_screen(self):
        self.clear_screen()
        self.current_screen = "login"
        self.liveness_confirmed = False
        self.liveness_detector = LivenessDetector()
        self.failed_attempts = 0

        main_frame = ctk.CTkFrame(self, fg_color="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(main_frame, fg_color="#16213e", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkButton(header, text="← Back",
                     font=("Arial", 14),
                     width=100, height=40,
                     fg_color="#333333",
                     command=self.show_home_screen).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(header, text="🔐 Login with Face Recognition",
                    font=("Arial", 22, "bold"),
                    text_color="#00d4ff").pack(side="left", padx=20)

        # Content
        content = ctk.CTkFrame(main_frame, fg_color="#1a1a2e")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Camera
        self.camera_label = ctk.CTkLabel(content, text="", width=600, height=450)
        self.camera_label.pack(side="left", padx=20)

        # Right panel
        right = ctk.CTkFrame(content, fg_color="#16213e", width=320)
        right.pack(side="right", fill="y", padx=20)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Authentication",
                    font=("Arial", 20, "bold"),
                    text_color="#ffffff").pack(pady=(40, 20))

        self.step1_label = ctk.CTkLabel(right,
                                        text="Step 1: Liveness Check",
                                        font=("Arial", 16, "bold"),
                                        text_color="#00d4ff")
        self.step1_label.pack(pady=5)

        self.liveness_status = ctk.CTkLabel(right,
                                            text="⏳ Waiting...",
                                            font=("Arial", 14),
                                            text_color="#888888")
        self.liveness_status.pack(pady=5)

        ctk.CTkFrame(right, height=2, fg_color="#333333").pack(
            fill="x", padx=20, pady=15)

        self.step2_label = ctk.CTkLabel(right,
                                        text="Step 2: Face Recognition",
                                        font=("Arial", 16, "bold"),
                                        text_color="#888888")
        self.step2_label.pack(pady=5)

        self.recognition_status = ctk.CTkLabel(right,
                                               text="⏳ Waiting...",
                                               font=("Arial", 14),
                                               text_color="#888888")
        self.recognition_status.pack(pady=5)

        ctk.CTkFrame(right, height=2, fg_color="#333333").pack(
            fill="x", padx=20, pady=15)

        self.access_status = ctk.CTkLabel(right,
                                          text="🔒 LOCKED",
                                          font=("Arial", 24, "bold"),
                                          text_color="#ff4444")
        self.access_status.pack(pady=20)

        self.failed_label = ctk.CTkLabel(right,
                                         text="",
                                         font=("Arial", 13),
                                         text_color="#ff4444")
        self.failed_label.pack(pady=5)

        # Start camera and login thread
        self.start_camera()
        threading.Thread(target=self.login_thread, daemon=True).start()

    def login_thread(self):
        frame_count = 0
        while self.current_screen == "login":
            if self.current_frame is None:
                continue

            frame = self.current_frame.copy()
            frame_count += 1

            # Phase 1: Liveness
            if not self.liveness_confirmed:
                _, is_live, status = self.liveness_detector.check_liveness(frame)
                self.liveness_status.configure(
                    text=f"👁️ {self.liveness_detector.current_challenge}",
                    text_color="#00d4ff"
                )
                if is_live:
                    self.liveness_confirmed = True
                    self.liveness_status.configure(
                        text="✅ Liveness Confirmed!",
                        text_color="#00ff88"
                    )
                    self.step2_label.configure(text_color="#00d4ff")

            # Phase 2: Recognition
            elif frame_count % 30 == 0:
                cv2.imwrite("data/temp_login.jpg", frame)
                try:
                    processed = preprocess_image(frame.copy())
                    cv2.imwrite("data/temp_login.jpg", processed)

                    result = DeepFace.represent(
                        img_path="data/temp_login.jpg",
                        model_name="Facenet",
                        enforce_detection=True
                    )

                    if result:
                        current_embedding = np.array(result[0]["embedding"])
                        best_match = None
                        best_distance = float('inf')

                        for name, stored_embedding in self.users:
                            distance = np.linalg.norm(
                                current_embedding - stored_embedding
                            )
                            if distance < 10.0 and distance < best_distance:
                                best_match = name
                                best_distance = distance

                        if best_match:
                            self.recognition_status.configure(
                                text=f"✅ {best_match} recognized!",
                                text_color="#00ff88"
                            )
                            self.access_status.configure(
                                text=f"🔓 WELCOME\n{best_match}!",
                                text_color="#00ff88"
                            )
                            self.log_access(best_match, "GRANTED")
                            self.failed_attempts = 0
                        else:
                            self.failed_attempts += 1
                            self.recognition_status.configure(
                                text="❌ Face not recognized!",
                                text_color="#ff4444"
                            )
                            self.failed_label.configure(
                                text=f"⚠️ Failed: {self.failed_attempts}/{self.max_failed}"
                            )
                            self.log_access("Unknown", "DENIED")

                            if self.failed_attempts >= self.max_failed:
                                self.access_status.configure(
                                    text="🚨 ALERT!\nToo many attempts!",
                                    text_color="#ff0000"
                                )
                                self.log_access("Unknown", "ALERT")

                except Exception:
                    self.recognition_status.configure(
                        text="👤 Looking for face...",
                        text_color="#888888"
                    )

    # ─────────────────────────────────────────
    # ADMIN SCREEN
    # ─────────────────────────────────────────
    def show_admin_screen(self):
        self.clear_screen()
        self.current_screen = "admin"

        main_frame = ctk.CTkFrame(self, fg_color="#1a1a2e")
        main_frame.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(main_frame, fg_color="#16213e", height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkButton(header, text="← Back",
                     font=("Arial", 14),
                     width=100, height=40,
                     fg_color="#333333",
                     command=self.show_home_screen).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(header, text="📊 Admin Dashboard",
                    font=("Arial", 22, "bold"),
                    text_color="#7c3aed").pack(side="left", padx=20)

        ctk.CTkButton(header, text="🔄 Refresh",
                     font=("Arial", 14),
                     width=100, height=40,
                     fg_color="#7c3aed",
                     command=self.show_admin_screen).pack(side="right", padx=20, pady=15)

        # Content
        content = ctk.CTkFrame(main_frame, fg_color="#1a1a2e")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Left - Users list
        left = ctk.CTkFrame(content, fg_color="#16213e", width=400)
        left.pack(side="left", fill="y", padx=10)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="👥 Registered Users",
                    font=("Arial", 18, "bold"),
                    text_color="#ffffff").pack(pady=20)

        # Users scrollable frame
        users_scroll = ctk.CTkScrollableFrame(left, fg_color="#1a1a2e",
                                              width=360, height=400)
        users_scroll.pack(padx=10, pady=10)

        try:
            conn = sqlite3.connect('data/database.db')
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, registered_at FROM users')
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                user_frame = ctk.CTkFrame(users_scroll, fg_color="#0f3460")
                user_frame.pack(fill="x", pady=5, padx=5)

                ctk.CTkLabel(user_frame,
                            text=f"#{row[0]} — {row[1]}",
                            font=("Arial", 14, "bold"),
                            text_color="#ffffff").pack(side="left", padx=10, pady=8)

                ctk.CTkLabel(user_frame,
                            text=f"{row[2][:10] if row[2] else 'N/A'}",
                            font=("Arial", 12),
                            text_color="#888888").pack(side="right", padx=10)

        except Exception as e:
            ctk.CTkLabel(users_scroll,
                        text=f"No users found",
                        text_color="#888888").pack(pady=20)

        # Right - Access logs
        right = ctk.CTkFrame(content, fg_color="#16213e")
        right.pack(side="right", fill="both", expand=True, padx=10)

        ctk.CTkLabel(right, text="📋 Recent Access Logs",
                    font=("Arial", 18, "bold"),
                    text_color="#ffffff").pack(pady=20)

        logs_scroll = ctk.CTkScrollableFrame(right, fg_color="#1a1a2e")
        logs_scroll.pack(fill="both", expand=True, padx=10, pady=10)

        log_path = "logs/access_logs.txt"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = f.readlines()

            for log in reversed(logs[-20:]):
                log = log.strip()
                if not log:
                    continue

                if "GRANTED" in log:
                    color = "#00ff88"
                elif "DENIED" in log:
                    color = "#ff4444"
                elif "ALERT" in log:
                    color = "#ff0000"
                else:
                    color = "#888888"

                ctk.CTkLabel(logs_scroll,
                            text=log,
                            font=("Arial", 12),
                            text_color=color,
                            wraplength=550,
                            anchor="w").pack(fill="x", pady=2, padx=5)
        else:
            ctk.CTkLabel(logs_scroll,
                        text="No logs found yet",
                        text_color="#888888").pack(pady=20)

    # ─────────────────────────────────────────
    # CAMERA FUNCTIONS
    # ─────────────────────────────────────────
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_running = True
        threading.Thread(target=self.camera_thread, daemon=True).start()

    def camera_thread(self):
        while self.camera_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (600, 450))
                    self.current_frame = frame.copy()

                    # Convert for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    img_tk = ImageTk.PhotoImage(image=img)

                    try:
                        self.camera_label.configure(image=img_tk)
                        self.camera_label.image = img_tk
                    except Exception:
                        break

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # ─────────────────────────────────────────
    # ACCESS LOGGER
    # ─────────────────────────────────────────
    def log_access(self, name, status):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] User: {name} | Status: {status}\n"
        with open("logs/access_logs.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)

    # ─────────────────────────────────────────
    # QUIT
    # ─────────────────────────────────────────
    def quit_app(self):
        self.stop_camera()
        self.quit()
        self.destroy()

# ─────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()