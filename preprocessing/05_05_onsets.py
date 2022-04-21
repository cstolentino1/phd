import numpy as np
import matplotlib.pyplot as plt
import sys
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks
from scipy.io.wavfile import write

sys.path.insert(0, '../utils')
import utils

# Extract audio from the video file
filename = '../data/video/phase5/05_05.mp4'
video = VideoFileClip(filename)
audio = video.audio
fs = audio.fps
signal = audio.to_soundarray(nbytes=4,buffersize=1000,fps=fs)

# Convert to mono and normalize
signal = signal.sum(axis=1) / 2
signal = signal/max(abs(signal))

# Compute spectral flux per frame
frameSize = 2048
hopLength = 441
spec_flux = utils.spectral_flux(signal, frameSize, hopLength)
delta = utils.calcThreshold(spec_flux, 7, 1, 2)

# Peak finding
peaks, _ = find_peaks(spec_flux, height = delta, distance = 80)
print("Number of peaks: ", len(peaks))
plt.plot(spec_flux)
plt.plot(peaks, spec_flux[peaks], "x")
plt.show()

# Convert peak locations to seconds
sample_onsets = (peaks*hopLength).astype(int)
time_onsets = (peaks*hopLength/fs)
np.save('../data/csv/phase5/05_05_time_onsets.npy', time_onsets)