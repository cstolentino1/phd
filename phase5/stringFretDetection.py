import numpy as np
import cv2
import sys
import collections
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from qtrangeslider import QRangeSlider

sys.path.insert(0, '../utils')
import utils

class StringFretDetection(QMainWindow):
    next_window = pyqtSignal()

    def __init__(self, cam_device_index, 
            init_str_theta, init_fr_theta, init_str_length, 
            init_fr_length, init_upper_edge, init_lower_edge, 
            init_left_edge, init_right_edge, line_threshold, 
            minLineLength, maxLineGap, lower_hsv1, upper_hsv1,
            lower_hsv2, upper_hsv2):

        super().__init__()
        self.cam_device_index = cam_device_index
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

        self.setWindowTitle('String Fret Detection')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.instructions_label = QLabel('Instructions: Double check if system can detect six strings and the two string ends.') 
        self.video_frame = VideoFrame(self)
        self.confirm_button = QPushButton('Confirm')
        
        self.layout = QVBoxLayout(self.central_widget)
        layout1 = QHBoxLayout()
        self.layout.addWidget(self.instructions_label)
        self.layout.addWidget(self.video_frame)
        self.layout.addLayout(layout1)
        layout1.addStretch()
        layout1.addWidget(self.confirm_button)
        self.setCentralWidget(self.central_widget)

        self.video_thread = VideoThread(self)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)
        self.video_thread.start()
        self.confirm_button.clicked.connect(self.confirm_button_clicked)
        self.video_thread.change_pixmap_signal.connect(self.video_frame.update_image)

    def confirm_button_clicked(self, s):
        confirm_dlg = QMessageBox.question(self, "Confirmation", "Confirm string fret detection?", QMessageBox.Yes | QMessageBox.No)
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
        self.annotations = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)
        self._run_flag = True
        self.line_threshold = self.main_window.line_threshold
        self.minLineLength = self.main_window.minLineLength
        self.maxLineGap = self.main_window.maxLineGap
        self.init_str_theta = self.main_window.init_str_theta
        self.init_fr_theta = self.main_window.init_fr_theta
        self.init_upper_edge = self.main_window.init_upper_edge
        self.init_lower_edge = self.main_window.init_lower_edge
        self.init_left_edge = self.main_window.init_left_edge
        self.init_right_edge = self.main_window.init_right_edge
        self.init_fr_length = self.main_window.init_fr_length
        self.init_str_length = self.main_window.init_str_length
        self.lower_hsv1 = self.main_window.lower_hsv1
        self.upper_hsv1 = self.main_window.upper_hsv1
        self.lower_hsv2 = self.main_window.lower_hsv2
        self.upper_hsv2 = self.main_window.upper_hsv2
        self.new_upper_edges = collections.deque(maxlen=10)
        self.new_lower_edges = collections.deque(maxlen=10)
        self.new_right_edges = collections.deque(maxlen=10)
        self.new_left_edges = collections.deque(maxlen=10)

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if not ret:
                continue
            cv_img = cv2.resize(cv_img, (self.frame_width, self.frame_height))
            line_image = np.copy(cv_img) * 0

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
            self.new_upper_edges.append(new_upper_edge)
            self.new_lower_edges.append(new_lower_edge)
            self.new_right_edges.append(new_right_edge)
            self.new_left_edges.append(new_left_edge)
            new_upper_edge = np.median(self.new_upper_edges).astype(int)
            new_lower_edge = np.median(self.new_lower_edges).astype(int)
            new_right_edge = np.median(self.new_right_edges).astype(int)
            new_left_edge = np.median(self.new_left_edges).astype(int)
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

            updated_frame = cv2.addWeighted(cv_img, 1, line_image, 1, 0)
            self.change_pixmap_signal.emit(updated_frame)

        self.cap.release()
    
    def stop(self):
        self._run_flag = False
        self.wait()

if __name__ == '__main__':
    app = QApplication([])
    window = StringFretDetection('../dissertation/data/videos/video18.mp4', 
        0.7828665232364396, -0.7977432152416729, 558.5892492750546, 
        57.277255868621296, 603.6202169287459, 660.8974727973672, 
        -105.02790735212929, 453.5613419229253, 100, 100, 20, 
        np.array([4, 13, 196], np.uint8), np.array([21, 38, 237], np.uint8),
        np.array([9, 34, 178], np.uint8), np.array([22, 56, 219], np.uint8))
    window.show()
    app.exit(app.exec())