import cv2
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QTextEdit
)

from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from gui.video_thread import VideoThread


class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Driver Monitoring System")
        self.resize(1250, 800)

        #GLOBAL STYLe
        self.setStyleSheet("""
        QWidget{
            background-color:#1e1e2f;
            color:white;
            font-family:Segoe UI;
        }
        """)

        self.thread = None

        
        title = QLabel("DRIVER MONITORING SYSTEM")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
        font-size:32px;
        font-weight:bold;
        color:#00e5ff;
        padding:10px;
        """)

        
        self.video_label = QLabel()
        self.video_label.setFixedSize(820, 520)
        self.video_label.setStyleSheet("""
        background:black;
        border:4px solid #00e5ff;
        border-radius:10px;
        """)

        
        self.head_label = QLabel("Head Direction : --")
        self.ear_label = QLabel("EAR : --")
        self.mar_label = QLabel("MAR : --")
        self.perclos_label = QLabel("PERCLOS : --")
        self.phone_label = QLabel("Phone : --")
        self.drowsy_label = QLabel("Drowsy : --")
        self.yawn_label = QLabel("Yawning : --")

        self.labels = [
            self.head_label,
            self.ear_label,
            self.mar_label,
            self.perclos_label,
            self.phone_label,
            self.drowsy_label,
            self.yawn_label
        ]

        for label in self.labels:
            label.setStyleSheet("""
            font-size:18px;
            padding:10px;
            background-color:#2b2b40;
            border-radius:8px;
            """)

        #STATUS PANEL
        status_layout = QVBoxLayout()
        for label in self.labels:
            status_layout.addWidget(label)

        status_frame = QFrame()
        status_frame.setLayout(status_layout)
        status_frame.setFixedWidth(330)
        status_frame.setStyleSheet("""
        background-color:#25253a;
        border-radius:10px;
        padding:10px;
        """)

        
        self.start_btn = QPushButton("START SYSTEM")
        self.stop_btn = QPushButton("STOP SYSTEM")

        self.start_btn.clicked.connect(self.start_system)
        self.stop_btn.clicked.connect(self.stop_system)

        self.start_btn.setStyleSheet("""
        QPushButton{
            background-color:#00c853;
            color:white;
            font-size:18px;
            padding:12px;
            border-radius:10px;
        }
        QPushButton:hover{background-color:#00e676;}
        """)

        self.stop_btn.setStyleSheet("""
        QPushButton{
            background-color:#d50000;
            color:white;
            font-size:18px;
            padding:12px;
            border-radius:10px;
        }
        QPushButton:hover{background-color:#ff1744;}
        """)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        # LOGS
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        self.log_box.setStyleSheet("""
        background-color:#111;
        color:#00ffcc;
        font-size:14px;
        border-radius:8px;
        padding:8px;
        """)

        #LAYOUT
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()

        content_layout.addWidget(self.video_label)
        content_layout.addWidget(status_frame)

        main_layout.addWidget(title)
        main_layout.addLayout(content_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

        #STATE MEMORY
        self.last_states = {
            "phone": False,
            "drowsy": False,
            "yawning": False,
            "head": "Forward"
        }

    #LOG FUNCTION
    def add_log(self, message):
        time_stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{time_stamp}] {message}")

    
    def start_system(self):
        if self.thread is None:#start
            self.thread = VideoThread()
            self.thread.frame_signal.connect(self.update_frame)
            self.thread.status_signal.connect(self.update_status)
            self.thread.start()
            self.add_log("SYSTEM STARTED")

    
    def stop_system(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None    #stop
            self.video_label.clear()
            self.add_log("SYSTEM STOPPED")

    # ---------- FRAME ----------
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    #STATUS
    def update_status(self, status):

        self.head_label.setText(f"Head Direction : {status['head']}")

        if status["ear"] is not None:
            self.ear_label.setText(f"EAR : {status['ear']:.3f}")

        if status["mar"] is not None:
            self.mar_label.setText(f"MAR : {status['mar']:.3f}")

        if status["perclos"] is not None:
            self.perclos_label.setText(f"PERCLOS : {status['perclos']:.3f}")

        #COLORS
        self.phone_label.setText("Phone : DETECTED" if status["phone"] else "Phone : None")
        self.phone_label.setStyleSheet(
            "background-color:#ffc107;" if status["phone"] else "background-color:#2b2b40;"
        )

        self.drowsy_label.setText("Drowsy : YES" if status["drowsy"] else "Drowsy : NO")
        self.drowsy_label.setStyleSheet(
            "background-color:#ff1744;" if status["drowsy"] else "background-color:#2b2b40;"
        )

        self.yawn_label.setText("Yawning : YES" if status["yawning"] else "Yawning : NO")
        self.yawn_label.setStyleSheet(
            "background-color:#ff9100;" if status["yawning"] else "background-color:#2b2b40;"
        )

        #LOGGING
        if status["phone"] and not self.last_states["phone"]:
            self.add_log("PHONE DETECTED")

        if status["drowsy"] and not self.last_states["drowsy"]:
            self.add_log("DRIVER DROWSY")

        if status["yawning"] and not self.last_states["yawning"]:
            self.add_log("YAWNING DETECTED")

        if status["head"] != self.last_states["head"]:
            if status["head"] == "Left":
                self.add_log("LOOKING LEFT")
            elif status["head"] == "Right":
                self.add_log("LOOKING RIGHT")
            elif status["head"] == "Forward":
                self.add_log("LOOKING FORWARD")

        self.last_states = status.copy()
