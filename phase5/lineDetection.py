import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

sys.path.insert(0, '../utils')
import utils

class LineDetection(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self, cam_device_index, init_upper_edge, init_lower_edge, init_fr_length, init_str_theta):
        super().__init__()
        self.cam_device_index = cam_device_index
        self.frame_width = 1280
        self.frame_height = 720

        # Parameters Needed
        self.init_upper_edge = init_upper_edge
        self.init_lower_edge = init_lower_edge
        self.init_fr_length = init_fr_length
        self.init_str_theta = init_str_theta

        # Parameters to be passed.
        self.line_threshold = None
        self.minLineLength = None
        self.maxLineGap = None

        self.setWindowTitle('Line Detection')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: Set the parameters for line detection. The system should mostly detect the lines of the guitar strings. \n' +  
            'Threshold: Accumulator threshold parameter. Only those lines that get enough votes are returned. \n' + 
            'MinLineLength: Minimum line length. Line segments shorter than that are rejected. \n' + 
            'MaxLineGap: Maximum allowed gap between points on the same line to link them.')
        self.video_frame = VideoFrame(self)
        self.slider1 = Slider("Threshold", 0, 200, 100)
        self.slider2 = Slider("Minimum Line Length", 0, 200, 100)
        self.slider3 = Slider("Maximum Line Gap", 0, 50, 20)
        self.line_threshold = self.slider1.slider.value()
        self.minLineLength = self.slider2.slider.value()
        self.maxLineGap = self.slider3.slider.value()
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
        self.slider1.slider.valueChanged.connect(self.video_thread.update_threshold)
        self.slider2.slider.valueChanged.connect(self.video_thread.update_minLineLength)
        self.slider3.slider.valueChanged.connect(self.video_thread.update_maxLineGap)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm line detection parameters?", QMessageBox.Yes | QMessageBox.No)
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
        self.annotations = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self._run_flag = True
        self._annotation_flag = True
        self.line_threshold = self.main_window.line_threshold
        self.minLineLength = self.main_window.minLineLength
        self.maxLineGap = self.main_window.maxLineGap

    def update_threshold(self, threshold):
        self.line_threshold = threshold
        self.main_window.line_threshold = threshold
    
    def update_minLineLength(self, minLineLength):
        self.minLineLength = minLineLength
        self.main_window.minLineLength = minLineLength
    
    def update_maxLineGap(self, maxLineGap):
        self.maxLineGap = maxLineGap
        self.main_window.maxLineGap = maxLineGap

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue
            cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
            line_image = np.copy(cv_img) * 0
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
            edges = cv2.Canny(blur, 50, 150, apertureSize = 3)
            lines_xy = cv2.HoughLinesP(edges, rho = 1, theta = np.pi/180, 
                threshold = self.line_threshold, minLineLength = self.minLineLength, maxLineGap = self.maxLineGap)
            if lines_xy is None:
                continue
            lines_xy = lines_xy.reshape(-1, 4)

            # Convert cartesian to polar
            lines_plr = []
            for x1, y1, x2, y2 in lines_xy:
                rho, theta = utils.rec2polar(x1, y1, x2, y2)
                lines_plr.append([rho, theta])
            lines_plr = np.array(lines_plr)

            # Remove lines outside initial upper edge and initial lower edge
            boolArr = (lines_plr[:,0] > self.main_window.init_upper_edge - 0.5 * self.main_window.init_fr_length) & (lines_plr[:,0] < self.main_window.init_lower_edge + 0.5 * self.main_window.init_fr_length)
            lines_xy = lines_xy[boolArr]
            lines_plr = lines_plr[boolArr]

            # Detect lines from within 10 degrees of initial string theta
            boolArr = (lines_plr[:,1] > (self.main_window.init_str_theta - 10*np.pi/180)) & (lines_plr[:,1] < (self.main_window.init_str_theta + 10*np.pi/180))
            lines_xy = lines_xy[boolArr]
            lines_plr = lines_plr[boolArr]
                
            for x1, y1, x2, y2 in lines_xy:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            updated_frame = cv2.addWeighted(cv_img, 1, line_image, 1, 0)
            self.change_pixmap_signal.emit(updated_frame)
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
    window = LineDetection(0, 0, 0, 0, 0)
    window.show()
    app.exit(app.exec())