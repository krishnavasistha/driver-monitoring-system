import cv2
import pygame
from PySide6.QtCore import QThread, Signal

from drowsiness_detector import DrowsinessDetector
from phone_detector import PhoneDetector


class VideoThread(QThread):

    frame_signal = Signal(object)
    status_signal = Signal(dict)

    def __init__(self):

        super().__init__()

        # ---------- AI DETECTORS ----------
        self.drowsiness_detector = DrowsinessDetector()

        self.phone_detector = PhoneDetector(
            custom_model_path="runs/detect/train7/weights/best.pt",
            coco_model_path="yolo11n.pt",
            conf_thres=0.20,
        )

        self.running = True

        # ---------- SOUND SYSTEM ----------
        pygame.mixer.init()

        self.alarm_drowsy = pygame.mixer.Sound("sound/alert_drowsy.mp3")
        self.alarm_phone = pygame.mixer.Sound("sound/alert_phone.mp3")

        self.ch_drowsy = pygame.mixer.Channel(0)
        self.ch_phone = pygame.mixer.Channel(1)

    def run(self):

        # ---------- CAMERA ----------
        cap = cv2.VideoCapture(0)

        # ---------- FPS BOOST (2 line fix) ----------
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.setUseOptimized(True)

        phone_smooth = [False] * 10

        while self.running:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()

            h, w = frame.shape[:2]

            # ---------- DROWSINESS ----------
            frame, ear, mar, is_drowsy, is_yawning, mesh_points, perclos = \
                self.drowsiness_detector.analyze(frame)

            # ---------- HEAD POSE ----------
            head_direction = "Forward"

            if mesh_points is not None:

                nose_tip = mesh_points[1]

                if nose_tip[0] < w // 3:
                    head_direction = "Left"

                elif nose_tip[0] > w * 2 // 3:
                    head_direction = "Right"

            # ---------- PHONE DETECTION ----------
            phone_in_use = self.phone_detector.detect(original_frame)

            phone_smooth.pop(0)
            phone_smooth.append(phone_in_use)

            phone_final = sum(phone_smooth) >= 3

            # ---------- ALERT SYSTEM ----------

            # Drowsiness Alert
            if is_drowsy and not self.ch_drowsy.get_busy():
                self.ch_drowsy.play(self.alarm_drowsy, loops=-1)

            elif not is_drowsy:
                self.ch_drowsy.stop()

            # Phone Alert
            if phone_final and not self.ch_phone.get_busy():
                self.ch_phone.play(self.alarm_phone, loops=-1)

            elif not phone_final:
                self.ch_phone.stop()

            # ---------- STATUS DATA ----------
            status = {
                "ear": ear,
                "mar": mar,
                "perclos": perclos,
                "drowsy": is_drowsy,
                "yawning": is_yawning,
                "phone": phone_final,
                "head": head_direction
            }

            # Send data to GUI
            self.status_signal.emit(status)

            # Send frame to GUI
            self.frame_signal.emit(frame)

        cap.release()

    def stop(self):

        self.running = False

        # Stop sounds
        self.ch_drowsy.stop()
        self.ch_phone.stop()

        pygame.mixer.quit()

        self.wait()