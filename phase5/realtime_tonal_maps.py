import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
from scipy.io.wavfile import read
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '../utils')
import utils

my_data = np.genfromtxt('output_data/iqui/data.csv', delimiter = ',')
plucking_positions = my_data[1:, 0]
angles_of_attack = my_data[1:, 1]
spectral_centroids = my_data[1:, 2]
pitch_estimates = my_data[1:, 3]
labels = my_data[1:, 4]

p1_indices = np.where(pitch_estimates == 196)[0]
p2_indices = np.where(pitch_estimates == 247)[0]
p3_indices = np.where(pitch_estimates == 330)[0]

input1 = np.array(plucking_positions)
input2 = np.array(angles_of_attack)
labels = np.array(labels)
output = np.array(spectral_centroids)

sc = StandardScaler()
input1_1, input1_2, input1_3 = input1[p1_indices], input1[p2_indices], input1[p3_indices]
input2_1, input2_2, input2_3 = input2[p1_indices], input2[p2_indices], input2[p3_indices]
labels1, labels2, labels3 = labels[p1_indices], labels[p2_indices], labels[p3_indices]
output1, output2, output3 = sc.fit_transform(output[p1_indices].reshape(-1, 1)), sc.fit_transform(output[p2_indices].reshape(-1, 1)), sc.fit_transform(output[p3_indices].reshape(-1, 1))

input1 = np.append(input1_1, np.append(input1_2, input1_3))
input2 = np.append(input2_1, np.append(input2_2, input2_3))
labels = np.append(labels1, np.append(labels2, labels3))
output = np.append(output1, np.append(output2, output3))

# input1 = input1_3
# input2 = input2_3
# labels = labels3
# output = output3

plt.scatter(labels, output)
plt.title('Brightness profiles per parameter combination')
plt.xlabel('Parameter combination')
plt.ylabel('Brightness score')
plt.ylim([-5, 5])
plt.show()

np.savetxt('06_on.csv', list(zip(labels, output)), delimiter = ',')

for n in range(9):
    indices = np.where(labels == n + 1)[0]
    plt.scatter(input1[indices], input2[indices])
plt.xlabel('Predicted plucking point position (cm)')
plt.ylabel('Predicted angle of attack (degrees)')
plt.title('Gestural parameter estimation')
plt.xlim([0, 32])
plt.ylim([0, 65])
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.show()

x1, x2, y, results = utils.tonalMap(input1, input2, output)
plt.contourf(x1, x2, y, np.linspace(y.min(), y.max(), 5))
plt.colorbar()
plt.title('Tonal map')
plt.xlabel('Plucking point position (cm)')
plt.ylabel('Angle of attack (degrees)')
plt.show()