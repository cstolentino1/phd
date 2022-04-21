import numpy as np
import cv2
import sys
import collections
import pyaudio
import threading
import atexit
import mediapipe as mp
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QLCDNumber, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
# from mathutils.geometry import intersect_point_line
from qtrangeslider import QRangeSlider

sys.path.insert(0, '../utils')
import utils

class PlayMode(QMainWindow):
    next_window = pyqtSignal()

    # def __init__(self, cam_device_index, mic_device_index):
    def __init__(self, cam_device_index, mic_device_index,
            init_str_theta, init_fr_theta, init_str_length, 
            init_fr_length, init_upper_edge, init_lower_edge, 
            init_left_edge, init_right_edge, line_threshold, 
            minLineLength, maxLineGap, lower_hsv1, upper_hsv1,
            lower_hsv2, upper_hsv2, min_detection_confidence, 
            min_tracking_confidence):

        super().__init__()
        self.cam_device_index = cam_device_index
        self.mic_device_index = mic_device_index
        self.frame_width = 1280
        self.frame_height = 720

        # Parameters Needed.
        self.init_str_theta = init_str_theta
        self.init_fr_theta = init_fr_theta
        self.init_str_length = init_str_length
        self.init_fr_length = init_fr_length
        self.init_upper_edge = init_upper_edge
        self.init_lower_edge = init_lower_edge
        self.init_left_edge = init_left_edge
        self.init_right_edge = init_right_edge 

        self.line_threshold = line_threshold
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap

        self.lower_hsv1 = lower_hsv1
        self.upper_hsv1 = upper_hsv1
        self.lower_hsv2 = lower_hsv2
        self.upper_hsv2 = upper_hsv2

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Data to be saved
        self.recorded_signals = []
        self.plucking_positions = []
        self.angles_of_attack = []
        self.onset_frames = []
        self.execution_time = []

        self.setWindowTitle('Performance Mode')
        self.resize(1050, 1000)
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: The guitarist performs with different gestural parameters. \n' + 
        'At each onset detected, the system estimates the plucking point position and angle of attack. \n' + 
        'Collect at least ten samples per parameter combination.') 
        self.video_frame = VideoFrame(self, 1024, 576)
        self.main_figure = MplFigure(self)
        self.parameter_estimates = QGroupBox()
        self.lcd_PP = QLCDNumber()
        self.lcd_AA = QLCDNumber()
        gblayout = QVBoxLayout()
        gblayout.addWidget(QLabel('Plucking point position (cm)'))
        gblayout.addWidget(self.lcd_PP)
        gblayout.addWidget(QLabel('Angle of attack (degrees)'))
        gblayout.addWidget(self.lcd_AA)
        self.parameter_estimates.setLayout(gblayout)
        self.output_frame = VideoFrame(self, 640, 360)
        self.output_signal = MplFigure(self)
        self.label_onset = QLabel('Awaiting Onset.')
        self.confirm_button = QPushButton('Confirm')
        
        self.layout = QVBoxLayout(self.central_widget)
        main_layout = QHBoxLayout()
        main_layout1 = QVBoxLayout()
        main_layout2 = QVBoxLayout()
        layout1 = QHBoxLayout()
        layout1.addWidget(self.main_figure.canvas)
        layout1.addWidget(self.parameter_estimates)
        main_layout1.addWidget(self.video_frame)
        main_layout1.addLayout(layout1)
        main_layout2.addWidget(self.output_frame)
        main_layout2.addWidget(self.output_signal.canvas)
        layout2 = QHBoxLayout()
        layout2.addWidget(self.label_onset)
        layout2.addStretch()
        layout2.addWidget(self.confirm_button)
        main_layout2.addLayout(layout2)
        main_layout2.addStretch()
        main_layout.addLayout(main_layout1)
        main_layout.addLayout(main_layout2)
        self.layout.addWidget(self.instructions_label)
        self.layout.addLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.displayCurrentFrame)
        self.timer.start(100)
        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.change_output_frame.connect(self.output_frame.update_image)
        self.video_thread.start()
        self.confirm_button.clicked.connect(self.confirm_button_clicked)

        self.mic = MicrophoneRecorder(self)
        self.mic.start(self.mic_device_index)

        self.time_vect = np.arange(self.mic.chunksize, dtype = np.float32) / self.mic.rate * 1000
        self.ax = self.main_figure.figure.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.time_vect.max())
        self.ax.set_xlabel('Time (ms)', fontsize = 6)
        self.ax.set_ylabel('Amplitude', fontsize = 6)
        self.ax.set_title('Audio signal', fontsize = 6)
        self.line, = self.ax.plot(self.time_vect, np.zeros_like(self.time_vect))

        self.time_vect_2 = np.arange(24 * self.mic.chunksize, dtype = np.float32) / self.mic.rate * 1000
        self.ax_2 = self.output_signal.figure.add_subplot(111)
        self.ax_2.set_ylim(-1,1)
        self.ax_2.set_xlim(0, self.time_vect_2.max())
        self.ax_2.set_xlabel('Time (ms)', fontsize=6)
        self.ax_2.set_ylabel('Amplitude', fontsize=6)
        self.ax_2.set_title('Recorded signal', fontsize=6)
        self.line_2, = self.ax_2.plot(self.time_vect_2, np.ones_like(self.time_vect_2))

    def displayCurrentFrame(self):
        if len(self.mic.frames) > 1:
            current_frame = self.mic.frames[-1]
            self.line.set_data(self.time_vect, current_frame)
            self.main_figure.canvas.draw()

    def displayRecordedSignal(self, recorded_frames):
        recorded_signal = np.array(recorded_frames).flatten()
        self.line_2.set_data(self.time_vect_2, recorded_signal)
        self.output_signal.canvas.draw()
        self.recorded_signals.append(recorded_signal)
        # write('example.wav', 48000, recorded_signal)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation","Produce tonal maps?", QMessageBox.Yes | QMessageBox.No)
        if confirm_dlg == QMessageBox.Yes:
            self.next_window.emit()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.mic.close()
        event.accept()

class VideoFrame(QWidget):
    def __init__(self, parent, frame_width, frame_height):
        super().__init__()
        self.main_window = parent
        self.frame_width = frame_width
        self.frame_height = frame_height
        
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
    change_output_frame = pyqtSignal(np.ndarray)

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

        self.init_str_theta = self.main_window.init_str_theta
        self.init_fr_theta = self.main_window.init_fr_theta
        self.init_str_length = self.main_window.init_str_length
        self.init_fr_length = self.main_window.init_fr_length
        self.init_upper_edge = self.main_window.init_upper_edge
        self.init_lower_edge = self.main_window.init_lower_edge
        self.init_left_edge = self.main_window.init_left_edge
        self.init_right_edge = self.main_window.init_right_edge

        self.line_threshold = self.main_window.line_threshold
        self.minLineLength = self.main_window.minLineLength
        self.maxLineGap = self.main_window.maxLineGap

        self.lower_hsv1 = self.main_window.lower_hsv1
        self.upper_hsv1 = self.main_window.upper_hsv1
        self.lower_hsv2 = self.main_window.lower_hsv2
        self.upper_hsv2 = self.main_window.upper_hsv2

        self.min_detection_confidence = self.main_window.min_detection_confidence
        self.min_tracking_confidence = self.main_window.min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode = False,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
            max_num_hands = 2
        )

        self.recorded_frames = []
        self.recording = False

        self.output_video = cv2.VideoWriter('output_data/output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25.0, (int(self.cap.get(3)), int(self.cap.get(4))))
    def run(self):
        self.frame_number = 0
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue
            self.output_video.write(cv_img)
            self.cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
            if (self.frame_number % 60) == 0:
                image_hand = cv2.cvtColor(cv2.flip(self.cv_img, 1), cv2.COLOR_BGR2RGB)
                image_hand.flags.writeable = False
                results = self.hands.process(image_hand)
            self.frame_number = self.frame_number + 1
            if self.recording == True and len(self.recorded_frames) < 5:
                self.recorded_frames.append(self.cv_img)
            self.change_pixmap_signal.emit(self.cv_img)
        self.cap.release()
        self.output_video.release()

    def recordFrames(self):
        self.start_time = time.time()
        self.recording = True

    def estimateParameter(self):
        plucking_positions = []
        angles_of_attack = []
        new_upper_edges = []
        new_lower_edges = []
        new_right_edges = []
        new_left_edges = []
        for n in range(len(self.recorded_frames)):
            cv_img = self.recorded_frames[n]
            line_image = np.copy(cv_img) * 0
            hand_annotations = np.copy(cv_img) * 0

            # Hand annotation
            image_hand = cv2.cvtColor(cv2.flip(cv_img, 1), cv2.COLOR_BGR2RGB)
            image_hand.flags.writeable = False
            results = self.hands.process(image_hand)
            right_hand_index = None
            if results.multi_hand_landmarks:
                for index, handedness in enumerate(results.multi_handedness):
                    if (handedness.classification[0].label) == "Right":
                        right_hand_index = index

                fingers_tip = []
                for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(hand_annotations, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if index != right_hand_index:
                        continue
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        if idx == 5:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, self.frame_width, self.frame_height)
                            index_knuckle = (int(self.frame_width - normalized_landmark[0] - 1), int(normalized_landmark[1]))
                        elif idx == 17:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, self.frame_width, self.frame_height)
                            little_knuckle = (int(self.frame_width-normalized_landmark[0]-1), int(normalized_landmark[1]))
                        elif idx == 8:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, self.frame_width, self.frame_height)
                            index_tip = (int(self.frame_width-normalized_landmark[0]-1), int(normalized_landmark[1]))
                            fingers_tip.append(index_tip)
                        elif idx == 12:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, self.frame_width, self.frame_height)
                            middle_tip = (int(self.frame_width-normalized_landmark[0]-1), int(normalized_landmark[1]))
                            fingers_tip.append(middle_tip)
                        elif idx == 16:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, self.frame_width, self.frame_height)
                            ring_tip = (int(self.frame_width-normalized_landmark[0]-1), int(normalized_landmark[1]))
                            fingers_tip.append(ring_tip)

            if right_hand_index == None:
                continue
            
            fingers_tip = np.mean(fingers_tip, axis = 0)
            hand_annotations = cv2.flip(hand_annotations, 1)

            # Color detection
            image_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            mask_hsv1 = cv2.inRange(image_hsv, self.lower_hsv1, self.upper_hsv1)
            mask_color1 = cv2.bitwise_and(cv_img, cv_img, mask = mask_hsv1)
            mask_color1 = cv2.cvtColor(mask_color1, cv2.COLOR_BGR2GRAY)
            (thresh, mask_color1) = cv2.threshold(mask_color1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask_color1 = cv2.medianBlur(mask_color1, 3)
            mask_color1 = cv2.dilate(mask_color1, np.ones((3, 3), np.uint8), iterations = 1)
            mask_color1 = cv2.erode(mask_color1, np.ones((3, 3), np.uint8), iterations = 1)

            mask_hsv2 = cv2.inRange(image_hsv, self.lower_hsv2, self.upper_hsv2)
            mask_color2 = cv2.bitwise_and(cv_img, cv_img, mask = mask_hsv2)
            mask_color2 = cv2.cvtColor(mask_color2, cv2.COLOR_BGR2GRAY)
            (thresh, mask_color2) = cv2.threshold(mask_color2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask_color2 = cv2.medianBlur(mask_color2, 3)
            mask_color2 = cv2.dilate(mask_color2, np.ones((3, 3), np.uint8), iterations = 1)
            mask_color2 = cv2.erode(mask_color2, np.ones((3, 3), np.uint8), iterations = 1)

            # Line Detection
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
            boolArr = (lines_plr[:,0] > self.init_upper_edge - 0.5 * self.init_fr_length) & (lines_plr[:,0] < self.init_lower_edge + 0.5*self.init_fr_length)
            lines_xy = lines_xy[boolArr]
            lines_plr = lines_plr[boolArr]
            
            # Detect lines from within 10 degrees of initial string theta
            boolArr = (lines_plr[:,1] > (self.init_str_theta - 10*np.pi/180)) & (lines_plr[:,1] < (self.init_str_theta + 10*np.pi/180))
            lines_xy = lines_xy[boolArr]
            lines_plr = lines_plr[boolArr]

            if len(lines_xy) < 5:
                continue

            # Calculate new theta
            str_theta = np.median(lines_plr[:,1])
            fr_theta = str_theta - 90*np.pi/180 if str_theta > 0 else str_theta + 90*np.pi/180

            # Rotate frame
            angle = str_theta * 180/np.pi - 90
            image_center = tuple(np.array(cv_img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image_rotate = cv2.warpAffine(cv_img, rot_mat, cv_img.shape[1::-1], flags = cv2.INTER_LINEAR)
            gray_rotate = cv2.cvtColor(cv2.GaussianBlur(image_rotate, (3, 3), 0), cv2.COLOR_BGR2GRAY)
            mask_color_rotate1 = cv2.warpAffine(mask_color1, rot_mat, cv_img.shape[1::-1], flags = cv2.INTER_LINEAR)
            mask_color_rotate2 = cv2.warpAffine(mask_color2, rot_mat, cv_img.shape[1::-1], flags = cv2.INTER_LINEAR)

            # Horizontal Sobel Filter and Post-Processing
            grad_x = cv2.Sobel(gray_rotate, cv2.CV_16S, 
                1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            (thresh, abs_grad_x) = cv2.threshold(abs_grad_x, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            abs_grad_x = cv2.medianBlur(abs_grad_x, 3)
            abs_grad_x = cv2.dilate(abs_grad_x, np.ones((5, 5), np.uint8), iterations = 1)
            abs_grad_x = cv2.erode(abs_grad_x, np.ones((5, 5), np.uint8), iterations = 1)
            abs_grad_x = cv2.dilate(abs_grad_x, np.ones((9, 3), np.uint8), iterations = 1)
                
            # Apply Mask
            upper_edge_rotated, _ = utils.polar_rotated(self.init_upper_edge, str_theta, angle, rot_mat)
            lower_edge_rotated, _ = utils.polar_rotated(self.init_lower_edge, str_theta, angle, rot_mat)
            right_edge_rotated, _ = utils.polar_rotated(self.init_right_edge, fr_theta, angle, rot_mat)
            left_edge_rotated, _ = utils.polar_rotated(self.init_left_edge, fr_theta, angle, rot_mat)
            min_y, max_y = max(int(upper_edge_rotated - 0.5 * self.init_fr_length), 0), min(int(lower_edge_rotated + 0.5 * self.init_fr_length), self.frame_height)
            min_x, max_x = max(int(left_edge_rotated - 0.20 * self.init_str_length), 0), min(int(right_edge_rotated + 0.20 * self.init_str_length), self.frame_width)
            
            mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
            mask[min_y: max_y, min_x: max_x] = 255
            abs_grad_x_masked = cv2.bitwise_and(abs_grad_x, abs_grad_x, mask = mask)
            abs_grad_x_binary = (abs_grad_x_masked > 0).astype(int)
            mask_color_rotate1 = cv2.bitwise_and(mask_color_rotate1, mask_color_rotate1, mask = mask)
            mask_color_rotate1 = (mask_color_rotate1 / 255).astype(int)
            mask_color_rotate2 = cv2.bitwise_and(mask_color_rotate2, mask_color_rotate2, mask = mask)
            mask_color_rotate2 = (mask_color_rotate2 / 255).astype(int)

            # Fret Detection
            fret_candidates = []
            start_cols = []
            end_cols = []
            num_white1 = []
            num_white2 = []
            upper_edges = []
            lower_edges = []
            flag = 0
            sum_white1 = 0
            sum_white2 = 0
            for col in range(min_x, max_x - 1):
                v_edge = abs_grad_x_binary[:, col]
                row = min_y
                current_height = 0
                while row < max_y - 1:
                    if v_edge[row] == 1:
                        start_row = row
                        while v_edge[row] == 1:
                            row = row + 1
                        end_row = row
                        height = end_row - start_row
                        if height > current_height:
                            current_height = height
                            lower = end_row
                            upper = start_row
                    else:
                        row = row + 1
                if (current_height > 0.50 * self.init_fr_length):
                    fret_candidates.append(col)
                    upper_edges.append(upper)
                    lower_edges.append(lower)
                    sum_white1 = sum_white1 + sum(mask_color_rotate1[:, col])
                    sum_white2 = sum_white2 + sum(mask_color_rotate2[:, col])
                    if flag == 0:
                        start_col = col
                        flag = 1
                else:
                    if flag == 1:
                        end_col = col
                        flag = 0
                        start_cols.append(start_col)
                        end_cols.append(end_col)
                        num_white1.append(sum_white1)
                        num_white2.append(sum_white2)
                        sum_white1 = 0
                        sum_white2 = 0
            
            if len(end_cols) < 5:
                continue

            # Boundaries
            new_upper_edge = np.median(upper_edges).astype(int)
            new_lower_edge = np.median(lower_edges).astype(int)
            left_side = end_cols[:len(end_cols) // 2]
            right_side = start_cols[len(start_cols)//2:]
            new_right_edge = right_side[np.argmax(num_white1[len(start_cols) // 2:])]
            left_edge_candidate1 = left_side[np.argmax(num_white2[:len(end_cols) // 2])]
            left_edge_candidate2 = int(new_right_edge - self.init_str_length)
            new_left_edge = left_edge_candidate1 if left_edge_candidate1 - 0.05 * self.init_str_length < left_edge_candidate2 < left_edge_candidate1 + 0.05 * self.init_str_length else left_edge_candidate2
            new_upper_edges.append(new_upper_edge)
            new_lower_edges.append(new_lower_edge)
            new_right_edges.append(new_right_edge)
            new_left_edges.append(new_left_edge)
            new_upper_edge = np.median(new_upper_edges).astype(int)
            new_lower_edge = np.median(new_lower_edges).astype(int)
            new_right_edge = np.median(new_right_edges).astype(int)
            new_left_edge = np.median(new_left_edges).astype(int)
            mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
            mask[new_upper_edge: new_lower_edge, new_left_edge: new_right_edge] = 255

            # String Mask
            strings = np.linspace(new_upper_edge, new_lower_edge, 8)[1:-1].astype(int)
            string_mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
            for string in strings:
                string_mask[string - 1: string + 1, new_left_edge: new_right_edge] = 255
            red_image = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
            red_image[:] = (0, 0, 255)
            line_image1 = cv2.bitwise_and(red_image, red_image, mask = string_mask)
            
            # Fret Mask
            fret_mask = np.zeros((self.frame_height, self.frame_width), np.uint8)
            fret_mask[new_upper_edge: new_lower_edge, new_right_edge - 3: new_right_edge + 3] = 1
            fret_mask[new_upper_edge: new_lower_edge, new_left_edge - 3: new_left_edge + 3] = 1
            # for index in range(len(start_cols)):
            #     middle_col = int((start_cols[index] + end_cols[index]) / 2)
            #     if middle_col < new_right_edge and middle_col > new_left_edge + 0.4 * self.init_str_length:
            #         fret_mask[new_upper_edge: new_lower_edge, middle_col - 1: middle_col + 1] = 255
            green_image = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
            green_image[:] = (0, 255, 0)
            line_image2 = cv2.bitwise_and(green_image, green_image, mask = fret_mask)

            # Line Image
            line_image = cv2.addWeighted(line_image1, 1, line_image2, 1, 0)
            angle = 90 - str_theta * 180/np.pi
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            line_image = cv2.warpAffine(line_image, rot_mat, cv_img.shape[1::-1], flags = cv2.INTER_LINEAR)

            # Rotate Boundaries
            new_upper_edge, _ = utils.polar_rotated(new_upper_edge, 90*np.pi/180, angle, rot_mat)
            new_lower_edge, _ = utils.polar_rotated(new_lower_edge, 90*np.pi/180, angle, rot_mat)
            new_right_edge, _ = utils.polar_rotated(new_right_edge, 0, angle, rot_mat)
            new_left_edge, _ =  utils.polar_rotated(new_left_edge, 0, angle, rot_mat)

            # # Extract plucking position
            second_string = np.linspace(new_upper_edge, new_lower_edge, 8)[4]
            x1, y1, x2, y2 = utils.polar2recv2(second_string, str_theta, new_left_edge, fr_theta, new_right_edge, fr_theta)
            relative_plucking_position = utils.intersect_point_line(np.array(fingers_tip), np.array([x1, y1]), np.array([x2, y2]))
            plucking_position = relative_plucking_position * 64.0
            plucking_positions.append(plucking_position)

            # Extract angle of attack
            _, angle_knuckle = utils.rec2polar(index_knuckle[0], index_knuckle[1], little_knuckle[0], little_knuckle[1])
            angle_of_attack = (str_theta - angle_knuckle) * 180/np.pi
            angles_of_attack.append(angle_of_attack)

            # Final annotations
            annotations = cv2.addWeighted(line_image, 1, hand_annotations, 1, 0)
            updated_frame = cv2.addWeighted(cv_img, 1, annotations, 1, 0)
            self.change_output_frame.emit(updated_frame)

        PP = np.median(plucking_positions)
        AA = np.median(angles_of_attack)

        print('Plucking point position: ', PP)
        print('Angle of Attack: ', AA)
        self.main_window.label_onset.setText('Awaiting Onset.')
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        print("Time Elapsed: %.4f"%(self.time_elapsed))

        if np.isnan(PP) or np.isnan(AA):
             self.main_window.recorded_signals.pop()
        else:
            self.main_window.lcd_PP.display(PP)
            self.main_window.lcd_AA.display(AA)

            self.main_window.plucking_positions.append(PP)
            self.main_window.angles_of_attack.append(AA)
            self.main_window.execution_time.append(self.time_elapsed)

        self.recorded_frames = []
        self.recording = False

    def stop(self):
        self._run_flag = False
        self.wait()

class MicrophoneRecorder(object):
    def __init__(self, parent, rate = 44100, chunksize = 1024):
        self.main_window = parent
        self.video_thread = self.main_window.video_thread
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.lamb = 1.0
        self.alpha = 2.0
        self.lock = threading.Lock()
        self.stop = False
        self.frames = collections.deque(maxlen = 9)
        self.odf = collections.deque(maxlen = 9)
        self.recording = False
        self.recorded_frames = []
        self.stream = None
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        currFrame = np.fromstring(data, 'int16') / 32768.0
        prevFrame = self.frames[-1] if len(self.frames) > 0 else currFrame
        with self.lock:
            self.frames.append(currFrame)
            self.odf.append(utils.calcFlux(currFrame, prevFrame))
            if len(self.frames) >= 9 and self.recording == False:
                if utils.checkOnset(np.array(self.odf), self.lamb, self.alpha):
                    # print('Onset Detected.')
                    self.main_window.label_onset.setText('Processing Onset.')
                    self.recording = True
                    self.recorded_frames.append(prevFrame)
                    self.video_thread.recordFrames()
            elif self.recording == True:
                self.recorded_frames.append(prevFrame)
                if len(self.recorded_frames) >= 24:
                    self.recording = False
                    self.main_window.displayRecordedSignal(self.recorded_frames)
                    self.video_thread.estimateParameter()
                    self.recorded_frames = []
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

class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white', figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)

if __name__ == '__main__':
    app = QApplication([])
    window = PlayMode('../data/video/phase5/05_01.mp4', 
        2, 0.7828665232364396, -0.7977432152416729, 558.5892492750546, 
        57.277255868621296, 603.6202169287459, 660.8974727973672, 
        -105.02790735212929, 453.5613419229253, 100, 100, 20, 
        np.array([4, 13, 196], np.uint8), np.array([21, 38, 237], np.uint8),
        np.array([9, 34, 178], np.uint8), np.array([22, 56, 219], np.uint8), 0.5, 0.5) 
    window.show()
    app.exit(app.exec())