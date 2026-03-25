import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# ------------------------- Setup -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# EAR eye indices (mediapipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# EAR threshold & counter
EAR_THRESH = 0.25
CONSEC_FRAMES = 20
counter = 0
drowsy = False

# ------------------------- Utility Functions -------------------------

def get_ear(landmarks, eye_points, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_direction(landmarks, w, h):
    image_points = np.array([
        [landmarks[1].x * w, landmarks[1].y * h],     # Nose tip
        [landmarks[152].x * w, landmarks[152].y * h], # Chin
        [landmarks[263].x * w, landmarks[263].y * h], # Left eye outer
        [landmarks[33].x * w, landmarks[33].y * h],   # Right eye outer
        [landmarks[61].x * w, landmarks[61].y * h],   # Left mouth
        [landmarks[291].x * w, landmarks[291].y * h], # Right mouth
    ], dtype="double")

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ])

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)
    direction = "Forward"

    if success:
        rot_matrix, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rot_matrix, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]

        if yaw < -15:
            direction = "Left"
        elif yaw > 15:
            direction = "Right"
        elif pitch < -15:
            direction = "Up"
        elif pitch > 15:
            direction = "Down"

    return direction

# ------------------------- Main Loop -------------------------

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    direction = "Unknown"
    EAR = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Head direction
        direction = get_head_direction(landmarks, w, h)

        # EAR calculation
        left_ear = get_ear(landmarks, LEFT_EYE, w, h)
        right_ear = get_ear(landmarks, RIGHT_EYE, w, h)
        EAR = (left_ear + right_ear) / 2.0

        # Drowsiness logic
        if EAR < EAR_THRESH:
            counter += 1
            if counter >= CONSEC_FRAMES:
                drowsy = True
        else:
            counter = 0
            drowsy = False

        # Face box
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        x_vals, y_vals = zip(*coords)
        cv2.rectangle(frame, (min(x_vals), min(y_vals)), (max(x_vals), max(y_vals)), (0, 255, 255), 2)

        # Eye keypoints
        for idx in LEFT_EYE + RIGHT_EYE:
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    # ------------------- Display -------------------
    cv2.putText(frame, f'Head: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'EAR: {EAR:.2f}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'FRAMES: {counter}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if drowsy:
        cv2.putText(frame, "DROWSY!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ------------------- Show Frame -------------------
    cv2.imshow('Head Pose & Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
