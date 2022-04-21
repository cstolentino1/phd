import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
import sys
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QComboBox, QDesktopWidget, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QLabel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '../utils')
import utils

class TonalMap(QMainWindow):

    def __init__(self, plucking_positions, angles_of_attack, spectral_centroids, pitch_estimates):

        super().__init__()
        self.plucking_positions = np.array(plucking_positions)
        self.angles_of_attack = np.array(angles_of_attack)
        self.spectral_centroid = np.array(spectral_centroids)
        self.pitch_estimates = np.array(pitch_estimates)
        self.sampling_rate = 48000
        self.pitch_candidates = np.array([196, 247, 330])

        self.colorbar = None
        self.plot_mode = 0
        self.regression_score = None

        self.setWindowTitle('Tonal Map')
        qr = self.frameGeometry()
        tp = QDesktopWidget().availableGeometry().topLeft()
        qr.moveTopLeft(tp)
        self.move(qr.topLeft())

        self.central_widget = QWidget()
        self.main_figure = MplFigure(self)
        self.regression_label = QLabel('Regression score: %s'%str(self.regression_score))
        self.pitch_box = QComboBox()
        self.mode_box = QComboBox()

        self.layout = QVBoxLayout(self.central_widget)
        hbox = QHBoxLayout()
        self.layout.addWidget(self.main_figure.canvas)
        hbox.addWidget(self.regression_label)
        hbox.addStretch()
        hbox.addWidget(self.pitch_box)
        hbox.addWidget(self.mode_box)
        self.layout.addLayout(hbox)
        self.setCentralWidget(self.central_widget)

        self.pitch_box.addItems(['G3 (196 Hz)', 'B3 (247 Hz)', 'E4 (330 Hz)', 'Combined'])
        self.pitch_box.currentIndexChanged.connect(self.chooseTonalMap)
        self.mode_box.addItems(['continuous', 'discrete'])
        self.mode_box.currentIndexChanged.connect(self.choosePlotMode)

        self.ax = self.main_figure.figure.add_subplot(111)
        self.ax.set_xlabel('Plucking positions (cm)', fontsize = 6)
        self.ax.set_ylabel('Angle of attack (degrees)', fontsize = 6)
        self.ax.set_title('Tonal map', fontsize = 6)

        p1_indices = np.where(self.pitch_estimates == self.pitch_candidates[0])[0]
        p2_indices = np.where(self.pitch_estimates == self.pitch_candidates[1])[0]
        p3_indices = np.where(self.pitch_estimates == self.pitch_candidates[2])[0]

        sc = StandardScaler()
        input1_1, input1_2, input1_3 = self.plucking_positions[p1_indices], self.plucking_positions[p2_indices], self.plucking_positions[p3_indices]
        input2_1, input2_2, input2_3 = self.angles_of_attack[p1_indices], self.angles_of_attack[p2_indices], self.angles_of_attack[p3_indices]
        output1, output2, output3 = sc.fit_transform(self.spectral_centroid[p1_indices].reshape(-1, 1)), sc.fit_transform(self.spectral_centroid[p2_indices].reshape(-1, 1)), sc.fit_transform(self.spectral_centroid[p3_indices].reshape(-1, 1))
        # output1, output2, output3 = self.spectral_centroid[p1_indices], self.spectral_centroid[p2_indices], self.spectral_centroid[p3_indices]
        input1 = np.append(input1_1, np.append(input1_2, input1_3))
        input2 = np.append(input2_1, np.append(input2_2, input2_3))
        output = np.append(output1, np.append(output2, output3))

        self.x1_1, self.x2_1, self.y_1, self.results_1 = utils.tonalMap(input1_1, input2_1, output1.reshape(-1))
        self.x1_2, self.x2_2, self.y_2, self.results_2 = utils.tonalMap(input1_2, input2_2, output2.reshape(-1))
        self.x1_3, self.x2_3, self.y_3, self.results_3 = utils.tonalMap(input1_3, input2_3, output3.reshape(-1))
        self.x1, self.x2, self.y, self.results = utils.tonalMap(input1, input2, output.reshape(-1))
        
        self.chooseTonalMap(None)

    def choosePlotMode(self, index):
        self.plot_mode = self.mode_box.currentIndex()
        self.chooseTonalMap(None)

    def chooseTonalMap(self, index):
        
        # continuous plot
        if self.plot_mode == 0:
            if self.pitch_box.currentIndex() == 0:
                img = self.ax.contourf(self.x1_1, self.x2_1, self.y_1, 200)
                self.regression_score = self.results_1.rsquared_adj
            elif self.pitch_box.currentIndex() == 1:
                img = self.ax.contourf(self.x1_2, self.x2_2, self.y_2, 200)
                self.regression_score = self.results_2.rsquared_adj
            elif self.pitch_box.currentIndex() == 2:
                img = self.ax.contourf(self.x1_3, self.x2_3, self.y_3, 200)
                self.regression_score = self.results_3.rsquared_adj
            elif self.pitch_box.currentIndex() == 3:
                img = self.ax.contourf(self.x1, self.x2, self.y, 200)
                self.regression_score = self.results.rsquared_adj
            if self.colorbar is not None:
                self.colorbar.remove()
            self.colorbar = self.main_figure.figure.colorbar(img)
            self.main_figure.canvas.draw()
            self.regression_label.setText('Regression score: %.4f'%(self.regression_score))


        # discrete plot        
        elif self.plot_mode == 1:
            if self.pitch_box.currentIndex() == 0:
                img = self.ax.contourf(self.x1_1, self.x2_1, self.y_1, np.linspace(self.y_1.min(), self.y_1.max(), 5))
                self.regression_score = self.results_1.rsquared_adj
            elif self.pitch_box.currentIndex() == 1:
                img = self.ax.contourf(self.x1_2, self.x2_2, self.y_2, np.linspace(self.y_2.min(), self.y_2.max(), 5))
                self.regression_score = self.results_2.rsquared_adj
            elif self.pitch_box.currentIndex() == 2:
                img = self.ax.contourf(self.x1_3, self.x2_3, self.y_3, np.linspace(self.y_3.min(), self.y_3.max(), 5))
                self.regression_score = self.results_3.rsquared_adj
            elif self.pitch_box.currentIndex() == 3:
                img = self.ax.contourf(self.x1, self.x2, self.y, np.linspace(self.y.min(), self.y.max(), 5))
                self.regression_score = self.results.rsquared_adj
            if self.colorbar is not None:
                self.colorbar.remove()
            self.colorbar = self.main_figure.figure.colorbar(img)
            self.main_figure.canvas.draw()
            self.regression_label.setText('Regression score: %.4f'%(self.regression_score))

class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white', figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)

if __name__ == '__main__':
    app = QApplication([])
    my_data = np.genfromtxt('../../dissertation/pedagogical_system/output_data/cheenie/data.csv', delimiter = ',')
    plucking_positions = my_data[1:,0]
    angles_of_attack = my_data[1:, 1]
    spectral_centroids = my_data[1:, 2]
    pitch_estimates = my_data[1:, 3]
    window = TonalMap(plucking_positions, angles_of_attack, spectral_centroids, pitch_estimates)
    window.show()
    app.exit(app.exec())