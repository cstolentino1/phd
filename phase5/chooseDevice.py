import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import pyaudio
import threading
import atexit
from PyQt5.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class ChooseDevice(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frame_width = 800
        self.frame_height = 600
        self.mplfigsize = (100,100)
        self.video_thread = VideoThread(self)

        self.setWindowTitle('Choose Devices')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.choose_cam_box = QComboBox()
        self.video_frame = VideoFrame(self)
        self.choose_mic_box = QComboBox()
        self.main_figure = MplFigure(self)
        self.confirm_button = QPushButton('Confirm')

        self.layout = QHBoxLayout(self.central_widget)
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        vbox1.addWidget(self.choose_cam_box)
        vbox1.addWidget(self.video_frame)
        vbox2.addWidget(self.choose_mic_box)
        vbox2.addWidget(self.main_figure.canvas)
        vbox2.addLayout(hbox1)
        hbox1.addStretch()
        hbox1.addWidget(self.confirm_button)
        self.layout.addLayout(vbox1)
        self.layout.addLayout(vbox2)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.handleNewData)
        self.timer.start(100)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.start()
        self.choose_cam_box.currentIndexChanged.connect(self.chooseCamDevice)
        self.choose_mic_box.currentIndexChanged.connect(self.chooseMicDevice)
        self.confirm_button.clicked.connect(self.confirm_button_clicked)

        self.mic = MicrophoneRecorder()
        self.mic_device_names = [name for name in self.mic.mic_devices.values()]
        self.choose_mic_box.addItems(self.mic_device_names)
        self.time_vect = np.arange(self.mic.chunksize, dtype=np.float32) / self.mic.rate * 1000

        self.cam_device_names = [name for name in self.video_thread.cam_devices.values()]
        self.choose_cam_box.addItems(self.cam_device_names)
    
        self.ax = self.main_figure.figure.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.time_vect.max())
        self.ax.set_xlabel('Time (ms)', fontsize=6)
        self.ax.set_ylabel('Amplitude', fontsize=6)
        self.ax.set_title('Audio signal.', fontsize=6)
        self.line, = self.ax.plot(self.time_vect, np.ones_like(self.time_vect))

    def handleNewData(self):
        frames = self.mic.get_frames()

        if len(frames) > 0:
            current_frame = frames[-1]
            self.line.set_data(self.time_vect, current_frame)
            self.main_figure.canvas.draw()
    
    def chooseMicDevice(self, index):
        self.mic_device_index = self.mic.inv_mic_devices[self.choose_mic_box.currentText()]
        self.mic.start(self.mic_device_index)

    def chooseCamDevice(self, index):
        self.cam_device_index = self.video_thread.inv_cam_devices[self.choose_cam_box.currentText()]
        self.video_thread.change_cam(self.cam_device_index)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm Devices?", QMessageBox.Yes | QMessageBox.No)
        if confirm_dlg == QMessageBox.Yes:
            self.next_window.emit()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.mic.close()
        event.accept()

class VideoFrame(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.frame_width = parent.frame_width
        self.frame_height = parent.frame_height
        self.frame_label = QLabel(self)
        self.frame_label.resize(self.frame_width, self.frame_height)

        self.layout = QVBoxLayout()
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
        self.frame_height = parent.frame_height
        self.frame_width = parent.frame_width
        self.cam_devices = {}
        num = 0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                num = num + 1
                self.cam_devices[i] = "Camera " + str(num)
                cap.release()
        self.inv_cam_devices = {value:key for key, value in self.cam_devices.items()}
        self.cap = None
        self._run_flag = True

    def run(self):
        while self._run_flag:
            if self.cap is not None:
                ret, cv_img = self.cap.read()
                if not ret:
                    continue
                cv_img = cv2.resize(cv_img,(self.frame_width,self.frame_height))
                self.change_pixmap_signal.emit(cv_img)
        self.cap.release()
    
    def change_cam(self, device_index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def stop(self):
        self._run_flag = False
        self.wait()

class MplFigure(object):
    def __init__(self, parent):
        figsize = parent.mplfigsize
        self.figure = plt.figure(facecolor='white', figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)

class MicrophoneRecorder(object):
    def __init__(self, rate = 44100, chunksize = 1024):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        self.mic_devices = {}
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                self.mic_devices[i] = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
        self.inv_mic_devices = {value:key for key, value in self.mic_devices.items()}
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        self.stream = None
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'int16') / 32768.0
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue
    
    def get_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames
    
    def start(self, device_index):
        if self.stream is not None:
            self.stream.stop_stream
            self.stream.close()
        try: 
            self.stream = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunksize,
                stream_callback=self.new_frame)
            self.stream.start_stream()
        except:
            print('An error occurred.')

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    app = QApplication([])
    window = ChooseDevice()
    window.show()
    app.exit(app.exec())