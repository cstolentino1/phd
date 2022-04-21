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
X = []  # audio data
y = []  # string length

L = 64.0    # string length
f0 = 247.0  # pitch for all samples

dirpath = '../data/audio/phase1'
for (path, names, fnames) in os.walk(dirpath):
    for name in fnames:
        fpath = os.path.join(path, name)

        fs, data = wavfile.read(fpath)

        part = int(name.split('_')[1])
        if part == 2:
            continue

        file_no = name.split('_')[2]
        file_no = int(file_no.split('.')[0])

        pluck_pos = (int((file_no-1)/10) + 1)*2
        R = pluck_pos / L
        if R > 0.5:
            R = 1 - R
        
        file_number.append(file_no)
        X.append(data)
        y.append(R)

index = sorted(range(len(file_number)), key = lambda k: file_number[k])
X = np.array(X)[index]
y = np.array(y)[index]

# Prediction
pred = []
mae = []
duration = []

for n in range(len(X)):
    data = X[n]
    label = y[n]

    start = time.time()
    p = utils.predict_R(data, fs, f0)
    elapsed = time.time() - start
    # print('Time elapsed: ', elapsed)
    duration.append(elapsed)
    pred.append(p)
    mae.append(np.abs(label-p)*L)

mae = np.array(mae)
print('Time elapsed per sample: ', np.mean(duration))

# Regression Plots
plt.scatter(y, pred, color='r')
plt.plot(np.arange(0,0.6,0.1), np.arange(0,0.6,0.1), color = 'b')
plt.title('Scatter plot')
plt.xlabel('True value of R')
plt.ylabel('Predicted value of R')
plt.legend(['True parameter line', 'Data points'])
plt.show()

# Mean absolute error
print('Mean absolute error is ',np.mean(mae),' cm.')

# Error trend wrt plucking position
error = []
pp = []
for r in np.unique(y):
    indices = np.array(np.where(y == r)[0])
    mae_pp = np.mean(mae[indices])
    pp.append(r*64)
    error.append(mae_pp)

plt.stem(pp,error)
plt.title('Mean absolute error per plucking position')
plt.xlabel('Plucking position (cm)')
plt.ylabel('Mean absolute error in cm')
plt.show()