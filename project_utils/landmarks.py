import numpy as np

class EyeAspectRatio:
    def __init__(self):
        self.left_indices = [33, 160, 158, 133, 153, 144]
        self.right_indices = [362, 385, 387, 263, 373, 380]

    def get_eyes(self, landmarks, size):
        h, w = size
        left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in self.left_indices]
        right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in self.right_indices]
        return left_eye, right_eye
    
    def compute_ear(self, left, right):
        def ear(eye):
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (A + B) / (2.0 * C)
        
        left_ear = ear(left)
        right_ear = ear(right)
        return (left_ear + right_ear) / 2.0
