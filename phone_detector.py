import os
import cv2
import torch
from ultralytics import YOLO
from collections import deque
import warnings

# Suppress all unnecessary logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

class PhoneDetector:
    def __init__(
        self,
        custom_model_path='runs/detect/train7/weights/best.pt',
        coco_model_path='yolo11n.pt',
        conf_thres=0.20,
        iou_thres=0.45,
        smooth_frames=5,
        frame_skip=2,     # Skip every 2 frames to reduce lag
        imgsz=480     # Lower image size = faster inference
    ):
        # Disable Ultralytics internal verbosity
    

        # Load YOLO models (both custom + COCO)
        self.model_custom = YOLO(custom_model_path)
        self.model_coco = YOLO(coco_model_path)

        # ✅ Force both models to run on CPU only
        self.model_custom.to("cpu")
        self.model_coco.to("cpu")

        # Parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.detections = deque(maxlen=smooth_frames)

        # Cache class names for speed
        self.class_names_custom = self.model_custom.names
        self.class_names_coco = self.model_coco.names

    def detect(self, frame):
        """Detects phone using both YOLO models, smooths detections, and skips frames for speed."""
        self.frame_count += 1

        # Skip frames to improve FPS
        if self.frame_count % self.frame_skip != 0:
            return False

        # Convert frame to RGB for YOLO
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        phone_detected = False

        with torch.no_grad():
            # Run both models with minimal verbosity
            res_custom = self.model_custom(rgb, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False)
            res_coco = self.model_coco(rgb, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False)

        # Check detections from both models
        for res, names in [(res_custom, self.class_names_custom), (res_coco, self.class_names_coco)]:
            if len(res) == 0:
                continue
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    label = names[cls].lower()
                    if ("phone" in label or "cell phone" in label) and conf >= self.conf_thres:
                        phone_detected = True
                        break
            if phone_detected:
                break

        # Smooth results to reduce flickering
        self.detections.append(phone_detected)
        detection_ratio = sum(self.detections) / len(self.detections)
        return detection_ratio > 0.3  # Trigger only if 30% of last frames detected phone
