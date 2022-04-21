import numpy as np
import cv2
import sys
import collections
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from qtrangeslider import QRangeSlider

sys.path.insert(0, '../utils')
import utils

class HandDetection(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self, cam_device_index):

        super().__init__()
        self.cam_device_index = cam_device_index
        self.frame_width = 1280
        self.frame_height = 720

        # Parameters to be passed.
        self.min_detection_confidence = None
        self.min_tracking_confidence = None

        self.setWindowTitle('Hand Detection')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: Adjust the hand detection model parameters so that the system can capture the right hand. \n' +  
            'Minimum Detection Confidence: Minimum confidence value for the detection to be considered successful. \n' + 
            'Minimum Tracking Confidence: Minimum confidence value for the hand landmarks to be considered tracked successfully.') 
        self.video_frame = VideoFrame(self)
        self.slider1 = Slider("Minimum Detection Confidence", 25, 75, 50)
        self.slider2 = Slider("Minimum Tracking Confidence", 25, 75, 50)
        self.min_detection_confidence = self.slider1.slider.value() / 100.
        self.min_tracking_confidence = self.slider2.slider.value() / 100.
        self.confirm_button = QPushButton('Confirm')
        
        self.layout = QVBoxLayout(self.central_widget)
        layout1 = QVBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QHBoxLayout()
        layout1.addWidget(self.slider1)
        layout1.addWidget(self.slider2)
        layout2.addStretch()
        layout2.addWidget(self.confirm_button)
        layout3.addLayout(layout1)
        layout3.addLayout(layout2)
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.video_frame)
        self.layout.addLayout(layout3)
        self.setCentralWidget(self.central_widget)

        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.start()
        self.confirm_button.clicked.connect(self.confirm_button_clicked)
        self.slider1.slider.valueChanged.connect(self.video_thread.update_parameters)
        self.slider2.slider.valueChanged.connect(self.video_thread.update_parameters)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm hand detection parameters?", QMessageBox.Yes | QMessageBox.No)
        if confirm_dlg == QMessageBox.Yes:
            self.next_window.emit()

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

class VideoFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.main_window = parent
        self.frame_width = self.main_window.frame_width
        self.frame_height = self.main_window.frame_height
        
        self.frame_label = QLabel(self)
        self.frame_label.resize(self.frame_width, self.frame_height)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.frame_label)
        self.setLayout(self.layout)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.frame_label.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.frame_width, self.frame_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent):
        super().__init__()
        self.main_window = parent
        self.cam_device_index = self.main_window.cam_device_index
        self.frame_height = self.main_window.frame_height
        self.frame_width = self.main_window.frame_width
        self.cap = cv2.VideoCapture(self.cam_device_index, cv2.CAP_DSHOW)
        # self.cap = cv2.VideoCapture(self.cam_device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._run_flag = True
        self.min_detection_confidence = self.main_window.min_detection_confidence
        self.min_tracking_confidence = self.main_window.min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            max_num_hands=2
        )

    def update_parameters(self, s):
        self.min_detection_confidence = self.main_window.slider1.value() / 100.
        self.min_tracking_confidence = self.main_window.slider2.value() / 100.
        self.main_window.min_detection_confidence = self.min_detection_confidence
        self.main_window.min_tracking_confidence = self.min_tracking_confidence
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            max_num_hands=2
        )

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue
            cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
            hand_annotations = np.copy(cv_img) * 0
            image_hand = cv2.cvtColor(cv2.flip(cv_img, 1), cv2.COLOR_BGR2RGB)
            image_hand.flags.writeable = False
            results = self.hands.process(image_hand)
            if results.multi_hand_landmarks:
                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(
                        hand_annotations, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            hand_annotations = cv2.flip(hand_annotations, 1)
            annotated_image = cv2.addWeighted(cv_img, 1, hand_annotations, 1, 0)
            self.change_pixmap_signal.emit(annotated_image)
        self.cap.release()
    
    def stop(self):
        self._run_flag = False
        self.wait()

class Slider(QWidget):
    def __init__(self, name, min, max, default = min):
        super().__init__()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min, max)
        self.slider.setValue(default)
        self.label_minimum = QLabel(str(min), alignment=Qt.AlignLeft)
        self.label_name = QLabel(name, alignment=Qt.AlignCenter)
        self.label_maximum = QLabel(str(max), alignment=Qt.AlignRight)

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.vbox.addWidget(self.slider)
        self.vbox.addLayout(self.hbox)
        self.hbox.addWidget(self.label_minimum, alignment=Qt.AlignLeft)
        self.hbox.addWidget(self.label_name, alignment=Qt.AlignCenter)
        self.hbox.addWidget(self.label_maximum, alignment=Qt.AlignRight)
        self.setLayout(self.vbox)

if __name__ == '__main__':
    app = QApplication([])
    window = HandDetection('../dissertation/data/videos/video18.mp4')
    window.show()
    app.exit(app.exec())