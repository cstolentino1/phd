import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '../utils')
import utils

# Read audio files
file_number = []
dirpath = '../data/audio/phase4'

X = []
y = []
pitch_estimates = []
for (path, names, fnames) in os.walk(dirpath):
    for name in fnames:
        fpath = os.path.join(path, name)
        fs, data = read(fpath)

        file_no = name.split('.')[0]
        if file_no == '04':
            continue
        file_no = int(file_no.split('_')[1])
        file_number.append(file_no - 1)
        
        label = int((file_no - 1) / 30) + 1
        if label == 2:
            label = 4
        elif label == 3:
            label = 7
        elif label == 4:
            label = 2
        elif label == 6:
            label = 8
        elif label == 7: 
            label = 3
        elif label == 8:
            label = 6

        if ((file_no - 1) % 3) == 0:
            pitch = 196
        elif ((file_no - 1) % 3) == 1:
            pitch = 247
        else:
            pitch = 330
        
        X.append(utils.spectral_centroid(data, fs))
        y.append(label)
        pitch_estimates.append(pitch)

file_number = np.array(file_number)
X = np.array(X)
y = np.array(y)
pitch_estimates = np.array(pitch_estimates)

p1_indices = np.where(pitch_estimates == 196)[0]
p2_indices = np.where(pitch_estimates == 247)[0]
p3_indices = np.where(pitch_estimates == 330)[0]

sc = StandardScaler()
output1, output2, output3 = sc.fit_transform(X[p1_indices].reshape(-1, 1)), sc.fit_transform(X[p2_indices].reshape(-1, 1)), sc.fit_transform(X[p3_indices].reshape(-1, 1))
y1, y2, y3 = y[p1_indices], y[p2_indices], y[p3_indices]
output = np.append(output1, np.append(output2, output3))
y = np.append(y1, np.append(y2, y3))

plt.scatter(y1, output1)
plt.title('f0 = 196 Hz')
plt.xlabel('Parameter combination')
plt.ylabel('Brightness score')
plt.show()

plt.scatter(y2, output2)
plt.title('f0 = 247 Hz')
plt.xlabel('Parameter combination')
plt.ylabel('Brightness score')
plt.show()

plt.scatter(y3, output3)
plt.title('f0 = 330 Hz')
plt.xlabel('Parameter combination')
plt.ylabel('Brightness score')
plt.show()

plt.scatter(y, output)
plt.title('All pitches')
plt.xlabel('Parameter combination')
plt.ylabel('Brightness score')
plt.show()
