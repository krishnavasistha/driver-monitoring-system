import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class DrowsinessDetector:
    def __init__(self, ear_thresh_init=0.22, mar_thresh=0.6, min_closed_sec=0.7, perclos_window_sec=30,
                 ear_alpha=0.35, mar_alpha=0.3, mar_consec_frames=3, baseline_frames=50):

        # --- Configurable Parameters ---
        self.ear_thresh_init = ear_thresh_init
        self.mar_thresh = mar_thresh
        self.min_closed_sec = min_closed_sec
        self.perclos_window_sec = perclos_window_sec
        self.ear_alpha = ear_alpha
        self.mar_alpha = mar_alpha
        self.mar_consec_frames = mar_consec_frames
        self.baseline_frames = baseline_frames

        # --- Mediapipe FaceMesh Setup ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # --- Landmark Indexes ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 387, 385, 263, 380, 373]
        self.MOUTH = [78, 81, 13, 311, 308, 14]
        # For drawing contours (full set)
       # self.LEFT_EYE_DRAW = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
       # self.RIGHT_EYE_DRAW = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        #self.MOUTH_DRAW = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]  # closes loop

        # --- State Variables ---
        self.ear_ema = None
        self.mar_ema = None
        self.closed_flags = deque()
        self.closed_run = 0
        self.mar_flags = deque(maxlen=self.mar_consec_frames)

        # --- Dynamic Baseline ---
        self.ear_baseline = []
        self.dynamic_thresh = self.ear_thresh_init
        self.mar_baseline = []

        # --- New: Stability Enhancements ---
        self.face_missing_counter = 0
        self.max_face_missing_tolerance = 5  # resets after 5 missed detections
        self.last_status = {"drowsy": False, "yawning": False}

        # --- New: reopen / yawn suppression parameters ---
        self.reopen_frames = 3               # consecutive open frames to confirm reopen
        self.reopen_counter = 0
        self.yawn_suppress_frames = 6        # suppress EAR drowsy alarms for this many frames after a yawn
        self.yawn_suppress_counter = 0
    def draw_box(self, frame, mesh_points, indices, color):
        xs = [mesh_points[i][0] for i in indices]
        ys = [mesh_points[i][1] for i in indices]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


    def calculate_ear(self, landmarks, eye_indices, shape):
        """Calculate Eye Aspect Ratio (EAR)"""
        h, w = shape[:2]
        eye = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_indices])
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)  

    def calculate_mar(self, landmarks, shape):
        """Calculate Mouth Aspect Ratio (MAR)"""
        h, w = shape[:2]
        mouth_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in self.MOUTH])
        A = np.linalg.norm(mouth_points[1] - mouth_points[5])
        B = np.linalg.norm(mouth_points[2] - mouth_points[4])
        C = np.linalg.norm(mouth_points[0] - mouth_points[3])
        return (A + B) / (2.0 * C + 1e-6)
    #def draw_shape(self, frame, mesh_points, indices, color):
        #pts = np.array([mesh_points[i] for i in indices], dtype=np.int32)
        #cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    def analyze(self, frame, fps=30):
        """Main method for drowsiness & yawning detection"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = mar = perclos = None
        mesh_points = None
        is_drowsy = False
        is_yawning = False

        # --- If face not detected ---
        if not results.multi_face_landmarks:
            self.face_missing_counter += 1
            if self.face_missing_counter > self.max_face_missing_tolerance:
                self.closed_run = 0
                self.mar_flags.clear()
                self.reopen_counter = 0
                self.yawn_suppress_counter = 0
            return frame, None, None, self.last_status["drowsy"], self.last_status["yawning"], None, 0.0

        self.face_missing_counter = 0  # reset counter
        landmarks = results.multi_face_landmarks[0].landmark
        mesh_points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

        # --- EAR Calculation ---
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, frame.shape)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, frame.shape)
        ear_raw = (left_ear + right_ear) / 2.0
        self.ear_ema = ear_raw if self.ear_ema is None else (
            self.ear_alpha * ear_raw + (1 - self.ear_alpha) * self.ear_ema
        )
        ear = self.ear_ema

        # --- Dynamic EAR Baseline ---
        if len(self.ear_baseline) < self.baseline_frames:
            if ear > self.ear_thresh_init:  # only open-eye EAR
                self.ear_baseline.append(ear)
        else:
            avg_eye = np.mean(self.ear_baseline)
            self.dynamic_thresh = max(self.ear_thresh_init, avg_eye * 0.75)

        # --- MAR Calculation ---
        mar_raw = self.calculate_mar(landmarks, frame.shape)
        self.mar_ema = mar_raw if self.mar_ema is None else (
            self.mar_alpha * mar_raw + (1 - self.mar_alpha) * self.mar_ema
        )
        mar = self.mar_ema

        # --- Adaptive MAR Threshold ---
        if len(self.mar_baseline) < self.baseline_frames:
            self.mar_baseline.append(mar)
        else:
            avg_mar = np.mean(self.mar_baseline)
            # keep a floor so threshold isn't too low
            self.mar_thresh = max(0.5, avg_mar * 1.6)

        # ----- Yawning detection (use >= to detect robustly) -----
    # Eye color switching based on EAR
            # ------- Draw eye and mouth shapes (contours) -------
        eye_color = (0, 255, 0) if ear > self.dynamic_thresh else (0, 0, 255)

        # Mouth color switching based on MAR
        #mouth_color = (0, 255, 0) if mar < self.mar_thresh else (0, 0, 255)

        # Perfect smooth shapes
        #self.draw_shape(frame, mesh_points, self.LEFT_EYE_DRAW, eye_color)
        #self.draw_shape(frame, mesh_points, self.RIGHT_EYE_DRAW, eye_color)
        #self.draw_shape(frame, mesh_points, self.MOUTH_DRAW, mouth_color)


        # append whether current smoothed MAR exceeds adaptive threshold
        self.mar_flags.append(mar > self.mar_thresh)
        # robust detection: at least mar_consec_frames positives in the window
        is_yawning = sum(self.mar_flags) >= self.mar_consec_frames

        # if yawn detected, set yawn suppression counter (prevents EAR drowsy alarms)
        if is_yawning:
            self.yawn_suppress_counter = self.yawn_suppress_frames

        # ----- Drowsiness detection with reopen confirmation & yawn suppression -----
        # If we are currently suppressing because of a yawn, don't mark drowsy from EAR
        if self.yawn_suppress_counter > 0:
            # decrease suppress counter each frame
            self.yawn_suppress_counter -= 1
            # ensure we don't set drowsy while yawning
            is_drowsy = False
            # we still track closed_run but do not trigger final drowsy while suppressed
            if ear < self.dynamic_thresh:
                self.closed_run += 1
            else:
                # treat this as an opening frame but only count reopen confirmation below
                pass
        else:
            # normal drowsiness logic
            if ear < self.dynamic_thresh:
                self.closed_run += 1
                self.reopen_counter = 0  # reset reopen counter when eyes are closed
            else:
                # eyes appear open this frame — increment reopen counter
                self.reopen_counter += 1
                if self.reopen_counter >= self.reopen_frames:
                    # confirmed open -> reset closed_run and drowsy status
                    self.closed_run = 0
                    self.reopen_counter = 0

            min_closed_frames = max(1, int(self.min_closed_sec * fps))
            is_drowsy = self.closed_run >= min_closed_frames

        # --- PERCLOS Calculation ---
        min_closed_frames = max(1, int(self.min_closed_sec * fps))
        self.closed_flags.append(self.closed_run >= min_closed_frames)
        max_len = max(1, int(self.perclos_window_sec * fps))
        while len(self.closed_flags) > max_len:
            self.closed_flags.popleft()
        perclos = sum(self.closed_flags) / len(self.closed_flags) if self.closed_flags else 0.0

        # --- Save Last State for Stability ---
        self.last_status["drowsy"] = is_drowsy
        self.last_status["yawning"] = is_yawning

        return frame, ear, mar, is_drowsy, is_yawning, mesh_points, perclos

