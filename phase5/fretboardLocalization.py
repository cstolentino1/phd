import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

sys.path.insert(0, '../utils')
import utils

class FretboardLocalization(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self, cam_device_index):
        super().__init__()
        self.cam_device_index = cam_device_index
        self.frame_width = 1280
        self.frame_height = 720
        self.video_thread = VideoThread(self)

        # Parameters to be passed.
        self.init_str_length = None
        self.init_str_theta = None
        self.init_fr_length = None
        self.init_fr_theta = None
        self.init_upper_edge = None
        self.init_lower_edge = None
        self.init_left_edge = None
        self.init_right_edge = None

        self.setWindowTitle('Fretboard Localization')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: Click on the fretboard edges. \n' +  
            'Press \'r\' to repeat annotations. \n' + 
            'Press \'c\' to confirm localization.')
        self.video_frame = VideoFrame(self)
        self.check_label = QLabel('Status: Fretboard not yet localized.')
        self.confirm_button = QPushButton('Confirm')
        
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.video_frame)
        layout1 = QHBoxLayout()
        layout1.addWidget(self.check_label)
        layout1.addStretch()
        layout1.addWidget(self.confirm_button)
        self.layout.addLayout(layout1)
        self.setCentralWidget(self.central_widget)

        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.start()
        self.confirm_button.clicked.connect(self.confirm_button_clicked)

    def confirm_button_clicked(self, s):
        if self.video_thread._annotation_flag == True:
            error_dlg = QMessageBox.critical(self, "Error", "You have not localized your fretboard.", QMessageBox.Ok)
        elif self.video_thread._annotation_flag == False:
            confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm fretboard location?", QMessageBox.Yes | QMessageBox.No)
            if confirm_dlg == QMessageBox.Yes:
                self.next_window.emit()
            else:
                self.video_thread.repeat_annotations()

    def keyPressEvent(self, event):
        # 'c' key is pressed
        if (event.key() == 67):
            if len(self.video_thread.ref_point) >= 3:
                self.video_thread.confirm_annotations()
        # 'r' key is pressed
        if (event.key() == 82):
            self.video_thread.repeat_annotations()

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
            self.main_window.video_thread.set_annotations(posF)

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
        self.ref_point = []
        self.rect = []

    def set_annotations(self, posF):
        if len(self.ref_point) < 3:
            normalized_x, normalized_y = posF
            self.image_height, self.image_width, _ = self.annotations.shape
            x, y = utils.normalized_to_pixel_coordinates(normalized_x, normalized_y, self.image_width, self.image_height)
            self.ref_point.append((x, y))

            if len(self.ref_point) == 1:
                cv2.circle(self.annotations, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            elif len(self.ref_point) == 2:
                cv2.line(self.annotations, self.ref_point[0], self.ref_point[1], (0, 0, 255), 4)
            elif len(self.ref_point) == 3:
                self.rect = utils.tri2rect(self.ref_point[0:3])
                cv2.line(self.annotations, self.rect[0], self.rect[1], (0, 0, 255), 4)
                cv2.line(self.annotations, self.rect[0], self.rect[2], (0, 0, 255), 4)
                cv2.line(self.annotations, self.rect[1], self.rect[3], (0, 0, 255), 4)
                cv2.line(self.annotations, self.rect[2], self.rect[3], (0, 0, 255), 4)

    def repeat_annotations(self):
        self.annotations = np.zeros((self.image_height, self.image_width, 3), np.uint8)
        self.ref_point = []
        self._annotation_flag = True
        self.main_window.check_label.setText('Status: Fretboard not yet localized.')
   
    def confirm_annotations(self):
        self._annotation_flag = False
        str_theta, fr_theta, upper_edge, lower_edge, left_edge, right_edge, fr_length, str_length = utils.extract_params(self.rect)
        self.main_window.init_str_theta = str_theta
        self.main_window.init_str_length = str_length
        self.main_window.init_fr_theta = fr_theta
        self.main_window.init_fr_length = fr_length
        self.main_window.init_upper_edge = upper_edge
        self.main_window.init_lower_edge = lower_edge 
        self.main_window.init_left_edge = left_edge
        self.main_window.init_right_edge = right_edge
        self.main_window.check_label.setText('Status: Fretboard has been localized!')

    def run(self):
        while self._run_flag:
            if self.cap is not None:
                ret, cv_img = self.cap.read()
                if not ret:
                    continue
                cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
                self.cv_img = cv2.addWeighted(cv_img, 1, self.annotations, 1, 0)
                self.change_pixmap_signal.emit(self.cv_img)
        self.cap.release()
    
    def stop(self):
        self._run_flag = False
        self.wait()

if __name__ == '__main__':
    app = QApplication([])
    window = FretboardLocalization(0)
    window.show()
    app.exit(app.exec())