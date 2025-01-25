import sys
sys.path.append('./')
import time
import numpy as np
import cv2
import os 
import torch
import warnings
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QScrollArea
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO
from boxmot.trackers.ocsort.ocsort import OcSort as OCSORT
from util import *
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("License Plate Detection and Recognition")
        self.setGeometry(100, 100, 1280, 720)

        # Initialize YOLO, Tracker, and OCR
        self.model = YOLO("weights/license_plate_detector.pt").to(device="cuda" if torch.cuda.is_available() else 'cpu')
        self.tracker = OCSORT()

        # Video Capture
        self.video = 'src_videos/sample.mp4'
        self.cap = cv2.VideoCapture(self.video)

        # Video Display
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(960, 540)
        self.video_label.setStyleSheet("background-color: lightgray;")

        # Sidebar for license plate and text
        self.sidebar = QListWidget(self)
        self.sidebar.setFixedWidth(300)
        self.sidebar.setStyleSheet("font-size: 16px; color: black;")  
        self.sidebar.setStyleSheet("background-color: lightgray; font-size: 16px; color: black;")  
        # FPS display label
        self.fps_label = QLabel(self)
        self.fps_label.setAlignment(Qt.AlignCenter)

        # Resume Button
        self.toggle_button = QPushButton("Stop", self)
        self.toggle_button.setFixedSize(200, 40)  # Make the button larger
        self.toggle_button.setStyleSheet("background-color: #FF6347; font-size: 18px; color: white;")  # Set color and size
        self.toggle_button.clicked.connect(self.toggle_video)

        # Layout
        main_layout = QVBoxLayout()
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.sidebar)
        video_layout.setStretch(0, 3)  # 3 parts for the video
        video_layout.setStretch(1, 1)  # 1 part for the sidebar

        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.toggle_button, alignment=Qt.AlignCenter)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Tracking variables
        self.frame_number = 0
        self.min_track_frames = 30
        self.tracks = {}
        self.finished_tracks = {}
        self.highest_scores = {}

        # FPS Calculation
        self.last_time = time.time()
        self.frame_count = 0

        # Pause state
        self.is_paused = False

    def crop_image(self, frame, bbox):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

    def toggle_video(self):
        """Toggles between stopping and resuming the video."""
        if self.is_paused:
            self.timer.start(30)  # Resume the video
            self.toggle_button.setText("Stop")  # Change button text to 'Stop'
        else:
            self.timer.stop()  # Stop the video
            self.toggle_button.setText("Resume")  # Change button text to 'Resume'
        self.is_paused = not self.is_paused


    def update_frame(self):
        if self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        self.frame_number += 1
        frame_resized = cv2.resize(frame, (1280, 640))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().to("cuda" if torch.cuda.is_available() else "cpu") / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)

        # YOLO Inference
        results = self.model(frame_tensor)
        detections = [
            [int(coord) for coord in output[:4]] + [output[4], int(output[5])]
            for output in results[0].boxes.data.tolist() if output is not None
        ]
        track_ids = self.tracker.update(np.asarray(detections), frame_resized) if detections else []

        # Process Tracks
        active_track_ids = {int(track[4]) for track in track_ids}
        for track_id in list(self.tracks.keys()):
            if track_id not in active_track_ids:
                if len(self.tracks[track_id]) >= self.min_track_frames:
                    finished_track = self.tracks.pop(track_id)
                    # Display the best cropped image for the track
                    if track_id in self.highest_scores:
                        best_crop = self.highest_scores[track_id]['image']
                        best_text = self.highest_scores[track_id]['text']
                        self.display_license_plate(track_id, best_crop, best_text)
                else:
                    self.tracks.pop(track_id)
                self.highest_scores.pop(track_id, None)  # Cleanup highest score for the track

        # Update Tracks and Draw Bounding Boxes
        for track in track_ids:
            bbox, track_id, confidence = track[:4], int(track[4]), track[5]
            cropped_image = self.crop_image(frame_resized, bbox)
           
            license_plate_text, license_plate_text_score = read_license_plate_by_paddle(cropped_image)

            if track_id not in self.tracks:
                self.tracks[track_id] = []
            self.tracks[track_id].append({'bbox': bbox, 'confidence': confidence, 'frame': self.frame_number})

            if license_plate_text:
                if track_id not in self.highest_scores or license_plate_text_score > self.highest_scores[track_id]['score']:
                    self.highest_scores[track_id] = {
                        'text': license_plate_text,
                        'score': license_plate_text_score,
                        'image': cropped_image
                    }

            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Track {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame_resized, f"Fps : {fps}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)

        # FPS Calculation
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.last_time)
            self.last_time = current_time
            self.fps_label.setText(f"FPS: {fps:.2f}")
            self.frame_count = 0

        # Update Video Frame in QLabel
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def display_license_plate(self, track_id, plate_image, plate_text):
        resized_plate = cv2.resize(plate_image, (200, 140), interpolation=cv2.INTER_LINEAR)
        item = QListWidgetItem(f"Car {track_id}: {plate_text}")
        self.sidebar.addItem(item)

        plate_image = cv2.cvtColor(resized_plate, cv2.COLOR_BGR2RGB)
        qimage = QImage(plate_image.data, plate_image.shape[1], plate_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(100, 100, Qt.KeepAspectRatio)
        label = QLabel()
        label.setPixmap(pixmap)
        list_item = QListWidgetItem()
        self.sidebar.addItem(list_item)
        self.sidebar.setItemWidget(list_item, label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
