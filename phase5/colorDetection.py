import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from qtrangeslider import QRangeSlider

sys.path.insert(0, '../utils')
import utils

class ColorDetection(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self, cam_device_index):
        super().__init__()
        self.cam_device_index = cam_device_index
        self.frame_width = 1280
        self.frame_height = 720
        self.neck_adjustment = True

        # Parameters to be passed.
        self.lower_hsv = np.array([0, 0, 0], np.uint8)
        self.upper_hsv = np.array([179, 255, 255], np.uint8)

        self.lower_hsv1 = np.array([0, 0, 0], np.uint8)
        self.upper_hsv1 = np.array([179, 255, 255], np.uint8)
        self.lower_hsv2 = np.array([0, 0, 0], np.uint8)
        self.upper_hsv2 = np.array([179, 255, 255], np.uint8)

        self.setWindowTitle('Color Detection')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: Determine the color mask of the bridge and the neck by adjusting the HSV parameters for each. \n' +  
            'Click on any frame to initialize the HSV parameters. \n' + 
            'Fine tune the HSV parameters such that only the bridge or the neck is detected. \n' + 
            'Adjust HSV parameters for the neck first, and then the bridge.')
        self.video_frame = VideoFrame(self)
        self.slider1 = RangeSlider("Hue", 0, 179, (0,179))
        self.slider2 = RangeSlider("Saturation", 0, 255, (0,255))
        self.slider3 = RangeSlider("Value", 0, 255, (0,255))
        self.confirm_button = QPushButton('Confirm')
        
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.video_frame)
        layout1 = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout3 = QVBoxLayout()
        layout1.addWidget(self.slider1)
        layout1.addWidget(self.slider2)
        layout1.addWidget(self.slider3)
        layout2.addLayout(layout1)
        layout2.addLayout(layout3)
        layout3.addStretch()
        layout3.addWidget(self.confirm_button)
        self.layout.addLayout(layout2)
        self.setCentralWidget(self.central_widget)

        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.start()
        self.confirm_button.clicked.connect(self.confirm_button_clicked)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.slider1.slider.valueChanged.connect(self.video_thread.update_hue)
        self.slider2.slider.valueChanged.connect(self.video_thread.update_saturation)
        self.slider3.slider.valueChanged.connect(self.video_thread.update_value)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm color detection parameters?", QMessageBox.Yes | QMessageBox.No)
        if confirm_dlg == QMessageBox.Yes and self.neck_adjustment == True:
            self.lower_hsv1 = self.lower_hsv
            self.upper_hsv1 = self.upper_hsv
            self.lower_hsv = np.array([0, 0, 0], np.uint8)
            self.upper_hsv = np.array([179, 255, 255], np.uint8)
            self.slider1.slider.setValue((0, 179))
            self.slider2.slider.setValue((0, 255))
            self.slider3.slider.setValue((0, 255))
            self.neck_adjustment = False
        elif confirm_dlg == QMessageBox.Yes and self.neck_adjustment == False:
            self.lower_hsv2 = self.lower_hsv
            self.upper_hsv2 = self.upper_hsv

            print('Lower HSV Neck: ', self.lower_hsv1)
            print('Upper HSV Neck: ', self.upper_hsv1)
            print('Lower HSV Bridge: ', self.lower_hsv2)
            print('Upper HSV Bridge: ', self.upper_hsv2)
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            posF = (event.pos().x() / self.frame_width, event.pos().y() / self.frame_height)
            self.main_window.video_thread.update_parameters(posF)

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
        self.hue = (0, 179)
        self.saturation = (0, 255)
        self.value = (0, 255)

    def update_hue(self, hue):
        self.hue = hue
        self.main_window.lower_hsv[0] = hue[0]
        self.main_window.upper_hsv[0] = hue[1]
    
    def update_saturation(self, saturation):
        self.saturation = saturation
        self.main_window.lower_hsv[1] = saturation[0]
        self.main_window.upper_hsv[1] = saturation[1]
    
    def update_value(self, value):
        self.value = value
        self.main_window.lower_hsv[2] = value[0]
        self.main_window.upper_hsv[2] = value[1]

    def update_parameters(self, posF):
        normalized_x, normalized_y = posF
        x, y = utils.normalized_to_pixel_coordinates(normalized_x, normalized_y, self.frame_width, self.frame_height)
        image_hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        h, s, v = image_hsv[y, x]
        self.hue = (max(h - int(0.20 * 179), 0), min(h + int(0.20 * 179), 179))
        self.saturation = (max(s - int(0.20 * 255), 0), min(s + int(0.20 * 255), 255))
        self.value = (max(v - int(0.20 * 255), 0), min(v + int(0.20 * 255), 255))
        self.main_window.slider1.slider.setValue(self.hue)
        self.main_window.slider2.slider.setValue(self.saturation)
        self.main_window.slider3.slider.setValue(self.value)

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue
            self.cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
            image_hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([self.hue[0], self.saturation[0], self.value[0]], dtype="uint8")
            upper_hsv = np.array([self.hue[1], self.saturation[1], self.value[1]], dtype="uint8")
            mask_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
            mask_color = cv2.bitwise_and(self.cv_img, self.cv_img, mask = mask_hsv)
            mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
            (thresh, mask_color) = cv2.threshold(mask_color, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask_color = cv2.medianBlur(mask_color, 3)
            mask_color = cv2.dilate(mask_color, np.ones((3, 3), np.uint8), iterations = 1)
            mask_color = cv2.erode(mask_color, np.ones((3, 3), np.uint8), iterations = 1)
            masked_img = cv2.bitwise_and(self.cv_img, self.cv_img, mask = mask_color)
            self.change_pixmap_signal.emit(masked_img)
        self.cap.release()
    
    def stop(self):
        self._run_flag = False
        self.wait()

class RangeSlider(QWidget):
    def __init__(self, name, min, max, default):
        super().__init__()
        self.slider = QRangeSlider(Qt.Horizontal)
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
    cam_device_index = '../data/video/phase5/05_05.mp4'
    window = ColorDetection(cam_device_index)
    window.show()
    app.exit(app.exec())