import cv2
import numpy as np
import mediapipe as mp
import time
import random

# ─────────────────────────────────────────
# SETUP MEDIAPIPE FACE MESH
# Detects 468 facial landmarks
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices for MediaPipe
# Left eye
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Right eye
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# Nose tip for head movement
NOSE_TIP = 1

# ─────────────────────────────────────────
# CALCULATE EYE ASPECT RATIO (EAR)
# ─────────────────────────────────────────
def calculate_EAR(eye_landmarks, landmarks, image_width, image_height):
    # Get coordinates
    points = []
    for idx in eye_landmarks:
        lm = landmarks[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append((x, y))

    # Vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

    # Horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# ─────────────────────────────────────────
# LIVENESS DETECTOR CLASS
# ─────────────────────────────────────────
class LivenessDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Blink detection
        self.EAR_THRESHOLD = 0.20
        self.blink_count = 0
        self.eye_closed = False
        self.required_blinks = 2

        # Head movement
        self.challenges = ["BLINK TWICE", "TURN LEFT", "TURN RIGHT", "NOD HEAD"]
        self.current_challenge = None
        self.challenge_completed = False
        self.nose_start_x = None

        # Liveness result
        self.is_live = False
        self.start_time = time.time()
        self.time_limit = 10  # seconds

    def get_new_challenge(self):
        self.current_challenge = random.choice(self.challenges)
        self.blink_count = 0
        self.challenge_completed = False
        self.nose_start_x = None
        self.start_time = time.time()
        print(f"🎯 New challenge: {self.current_challenge}")

    def check_liveness(self, frame):
        if self.current_challenge is None:
            self.get_new_challenge()

        image_height, image_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "Checking liveness..."
        color = (0, 165, 255)  # Orange

        # Check time limit
        elapsed = time.time() - self.start_time
        remaining = max(0, self.time_limit - elapsed)

        if elapsed > self.time_limit and not self.challenge_completed:
            self.get_new_challenge()
            return frame, False, "Time up! New challenge..."

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ── BLINK DETECTION ──
            left_ear = calculate_EAR(
                LEFT_EYE_EAR, landmarks, image_width, image_height
            )
            right_ear = calculate_EAR(
                RIGHT_EYE_EAR, landmarks, image_width, image_height
            )
            avg_ear = (left_ear + right_ear) / 2.0

            # Detect blink
            if avg_ear < self.EAR_THRESHOLD:
                self.eye_closed = True
            elif self.eye_closed and avg_ear >= self.EAR_THRESHOLD:
                self.blink_count += 1
                self.eye_closed = False
                print(f"👁️ Blink detected! Count: {self.blink_count}")

            # ── NOSE POSITION FOR HEAD MOVEMENT ──
            nose = landmarks[NOSE_TIP]
            nose_x = nose.x  # Normalized 0-1

            if self.nose_start_x is None:
                self.nose_start_x = nose_x

            nose_movement = nose_x - self.nose_start_x

            # ── CHECK CHALLENGE ──
            if self.current_challenge == "BLINK TWICE":
                if self.blink_count >= 2:
                    self.challenge_completed = True
                    self.is_live = True
                status = f"👁️ BLINK TWICE! Blinks: {self.blink_count}/2 | Time: {remaining:.1f}s"
                color = (0, 255, 255)

            elif self.current_challenge == "TURN LEFT":
                if nose_movement > 0.05:
                    self.challenge_completed = True
                    self.is_live = True
                status = f"⬅️ TURN LEFT! | Time: {remaining:.1f}s"
                color = (255, 165, 0)

            elif self.current_challenge == "TURN RIGHT":
                if nose_movement < -0.05:
                    self.challenge_completed = True
                    self.is_live = True
                status = f"➡️ TURN RIGHT! | Time: {remaining:.1f}s"
                color = (255, 165, 0)

            elif self.current_challenge == "NOD HEAD":
                nose_y = nose.y
                if self.nose_start_x is not None:
                    if abs(nose_y - landmarks[NOSE_TIP].y) > 0.02:
                        self.challenge_completed = True
                        self.is_live = True
                status = f"⬆️ NOD HEAD! | Time: {remaining:.1f}s"
                color = (255, 165, 0)

            # Draw EAR value
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.challenge_completed:
                status = "✅ LIVENESS CONFIRMED! You are REAL!"
                color = (0, 255, 0)

        else:
            status = "⚠️ No face detected - move closer!"
            color = (0, 0, 255)

        # Display challenge on frame
        cv2.putText(frame, f"Challenge: {self.current_challenge}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, self.challenge_completed, status


# ─────────────────────────────────────────
# TEST LIVENESS DETECTION
# ─────────────────────────────────────────
def test_liveness():
    print("\n🔐 Liveness Detection Test")
    print("Complete the challenge shown on screen!")
    print("Press 'r' to reset | Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    detector = LivenessDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, is_live, status = detector.check_liveness(frame)

        if is_live:
            cv2.putText(frame, "ACCESS GRANTED!", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector = LivenessDetector()
            print("🔄 Reset!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_liveness()