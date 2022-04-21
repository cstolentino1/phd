import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.io import wavfile

sys.path.insert(0, '../utils')
import utils

# Read audio files
file_number = []
X = []      # audio data
f = []      # pitch data
y = []      # label data
L = []      # string length

fret_number = np.array([3, 5, 8])
string_length = 64*2**(-fret_number / 12)
pitch = 247.0*2**(fret_number / 12)
pluck_pos = [4, 10, 16, 22, 28]

dirpath = '../data/audio/phase1'
for (path, names, fnames) in os.walk(dirpath):
    for name in fnames:
        fpath = os.path.join(path,name)

        fs, data = wavfile.read(fpath)

        part = int(name.split('_')[1])
        if part == 1:
            continue
        
        file_no = name.split('_')[2]
        file_no = int(file_no.split('.')[0])
        
        i_pluck_pos = int((file_no-1) / 10) % 5
        i_strlen = int((file_no-1) / 50)
        i_pitch = int((file_no-1) / 50)

        f0 = pitch[i_pitch]
        R = pluck_pos[i_pluck_pos]/string_length[i_strlen]
        if R > 0.5:
            R = 1 - R

        file_number.append(file_no)
        X.append(data)
        f.append(f0)
        y.append(R)
        L.append(string_length[i_strlen])

index = sorted(range(len(file_number)), key = lambda k: file_number[k])
X = np.array(X)[index]
y = np.array(y)[index]
f = np.array(f)[index]
L = np.array(L)[index]

# Prediction
pred = []
mae = []
duration = []

for n in range(len(X)):
    data = X[n]
    f0 = f[n]
    label = y[n]
    str_len = L[n]

    start = time.time()
    p = utils.predict_R(data, fs, f0)
    elapsed = time.time() - start
    duration.append(elapsed)
    pred.append(p)
    mae.append(np.abs(label-p)*str_len)

print('Time elapsed per sample: ', np.mean(duration))

# Regression Plots
plt.scatter(y, pred, color='r')
plt.plot(np.arange(0, 0.6, 0.1), np.arange(0, 0.6, 0.1), color = 'b')
plt.title('Scatter plot')
plt.xlabel('True value of R')
plt.ylabel('Predicted value of R')
plt.legend(['True parameter line', 'Data points'])
plt.show()

# Mean absolute error
print('Mean absolute error is ',np.mean(mae),' cm.')

# Error trend wrt pitch
error = [np.mean(mae[0:50]), np.mean(mae[50:100]), np.mean(mae[100:150])]
plt.stem(pitch, error)
plt.title('Mean absolute error per pitch')
plt.xlabel('Plucking position (cm)')
plt.ylabel('Mean absolute error in cm')
plt.show()