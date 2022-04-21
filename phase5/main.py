import sys
import numpy as np
import csv
from PyQt5.QtWidgets import QApplication
from chooseDevice import ChooseDevice
from fretboardLocalization import FretboardLocalization
from lineDetection import LineDetection
from colorDetection import ColorDetection
from stringFretDetection import StringFretDetection
from handDetection import HandDetection
from playMode import PlayMode
from tonalMap import TonalMap

sys.path.insert(0, '../utils')
import utils

class Controller:

    def __init__(self):
        # Parameters
        self.sampling_rate = 44100
        self.cam_device_index = None
        self.mic_device_index = None

        self.init_str_theta = None
        self.init_fr_length = None
        self.init_fr_theta = None
        self.init_upper_edge = None
        self.init_lower_edge = None
        self.init_left_edge = None
        self.init_right_edge = None
        
        self.line_threshold = None
        self.minLineLength = None
        self.maxLineGap = None

        self.lower_hsv1 = None
        self.upper_hsv1 = None
        self.lower_hsv2 = None
        self.upper_hsv2 = None

        self.min_detection_confidence = None
        self.min_tracking_confidence = None

        self.model = None
        
    def ChooseDevice(self):
        self.choose_device = ChooseDevice()
        self.choose_device.next_window.connect(self.FretboardLocalization)
        self.choose_device.show()

    def FretboardLocalization(self):
        self.cam_device_index = self.choose_device.cam_device_index
        self.mic_device_index = self.choose_device.mic_device_index

        # self.cam_device_index = '../phd/data/video/phase1/01_01.mkv'
        self.choose_device.close()
        self.fretboard_localization = FretboardLocalization(self.cam_device_index)
        self.fretboard_localization.next_window.connect(self.LineDetection)
        self.fretboard_localization.show()

    def LineDetection(self):
        self.init_str_length = self.fretboard_localization.init_str_length
        self.init_str_theta = self.fretboard_localization.init_str_theta
        self.init_fr_length = self.fretboard_localization.init_fr_length
        self.init_fr_theta = self.fretboard_localization.init_fr_theta
        self.init_upper_edge = self.fretboard_localization.init_upper_edge
        self.init_lower_edge = self.fretboard_localization.init_lower_edge
        self.init_left_edge = self.fretboard_localization.init_left_edge
        self.init_right_edge = self.fretboard_localization.init_right_edge

        self.fretboard_localization.close()
        self.line_detection = LineDetection(self.cam_device_index, self.init_upper_edge, 
            self.init_lower_edge, self.init_fr_length, self.init_str_theta)
        self.line_detection.next_window.connect(self.ColorDetection)
        self.line_detection.show()

    def ColorDetection(self):
        self.line_threshold = self.line_detection.line_threshold
        self.minLineLength = self.line_detection.minLineLength
        self.maxLineGap = self.line_detection.maxLineGap

        self.line_detection.close()
        self.color_detection = ColorDetection(self.cam_device_index)
        self.color_detection.next_window.connect(self.StringFretDetection)
        self.color_detection.show()

    def StringFretDetection(self):
        self.lower_hsv1 = self.color_detection.lower_hsv1
        self.upper_hsv1 = self.color_detection.upper_hsv1
        self.lower_hsv2 = self.color_detection.lower_hsv2
        self.upper_hsv2 = self.color_detection.upper_hsv2

        self.color_detection.close()
        self.stringfret_detection = StringFretDetection(self.cam_device_index, self.init_str_theta, 
            self.init_fr_theta, self.init_str_length, self.init_fr_length, self.init_upper_edge, self.init_lower_edge, 
            self.init_left_edge, self.init_right_edge, self.line_threshold, self.minLineLength, self.maxLineGap, 
            self.lower_hsv1, self.upper_hsv1, self.lower_hsv2, self.upper_hsv2)
        self.stringfret_detection.next_window.connect(self.HandDetection)
        self.stringfret_detection.show()

    def HandDetection(self):
        self.stringfret_detection.close()
        self.hand_detection = HandDetection(self.cam_device_index)
        self.hand_detection.next_window.connect(self.PlayMode)
        self.hand_detection.show()

    def PlayMode(self):
        self.min_detection_confidence = self.hand_detection.min_detection_confidence
        self.min_tracking_confidence = self.hand_detection.min_tracking_confidence

        self.hand_detection.close()
        self.play_mode = PlayMode(self.cam_device_index, self.mic_device_index, self.init_str_theta, self.init_fr_theta, 
            self.init_str_length, self.init_fr_length, self.init_upper_edge, self.init_lower_edge, self.init_left_edge, 
            self.init_right_edge, self.line_threshold, self.minLineLength, self.maxLineGap, self.lower_hsv1, 
            self.upper_hsv1, self.lower_hsv2, self.upper_hsv2, self.min_detection_confidence, self.min_tracking_confidence)
        self.play_mode.next_window.connect(self.TonalMap)
        self.play_mode.show()

    def TonalMap(self):
        self.recorded_signals = self.play_mode.recorded_signals
        self.plucking_positions = self.play_mode.plucking_positions
        self.angles_of_attack = self.play_mode.angles_of_attack

        self.spectral_centroids = []
        self.pitch_estimates = []

        pitch_collection = np.hstack((98.00*2**np.arange(11), 123.47*2**np.arange(11), 82.41*2**np.arange(11)))
        pitch_candidates = [196, 247, 330]
        
        for n in range(len(self.recorded_signals)):
            signal = self.recorded_signals[n]
            freq, corr, ip = utils.freq_from_autocorr(signal, self.sampling_rate)
            pitch_diff = np.abs(pitch_collection - freq)
            idx = pitch_diff.argmin()
            if pitch_diff[idx] < 20:        # pitch difference must be less than 20 Hz
                pitch = pitch_candidates[int(idx / 11)]
            else:
                pitch = 0
            self.pitch_estimates.append(pitch)
            self.spectral_centroids.append(utils.spectral_centroid(signal, self.sampling_rate))
        self.pitch_estimates = np.array(self.pitch_estimates)
        self.spectral_centroids = np.array(self.spectral_centroids)
        self.data = zip(self.plucking_positions, self.angles_of_attack, self.spectral_centroids, self.pitch_estimates)

        np.save('output_data/recorded_signals.npy', self.recorded_signals)

        with open('output_data/data.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['plucking positions', 'angles of attack', 'spectral centroid', 'pitch estimate'])
    
            # write multiple rows
            writer.writerows(self.data)

        # my_data = np.genfromtxt('../dissertation/pedagogical_system/output_data/elkan/data.csv', delimiter = ',')
        # self.plucking_positions = my_data[1:,0]
        # self.angles_of_attack = my_data[1:, 1]
        # self.spectral_centroids = my_data[1:, 2]
        # self.pitch_estimates = my_data[1:, 3]
        
        self.play_mode.close()
        self.tonal_map = TonalMap( 
            self.plucking_positions, self.angles_of_attack, 
            self.spectral_centroids, self.pitch_estimates)
        self.tonal_map.show()

    def exit(self):
        self.play_mode.close()

if __name__ == '__main__':
    app = QApplication([])
    controller = Controller()
    controller.ChooseDevice()
    app.exit(app.exec())