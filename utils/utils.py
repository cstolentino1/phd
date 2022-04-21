import numpy as np
import math
import cv2
import peakutils
import statsmodels.api as sm
from scipy.signal import find_peaks, fftconvolve

def predict_R(x, fs, f0):
    
    # Fourier Transform
    X = np.fft.fft(x)
    X = X[:int(len(X)/2)]
    X = np.abs(X)
    X = X/np.max(X)
    f = np.linspace(0, fs/2, len(X))

    # Peak detection
    start = len(X) * ((0.8*f0) / (fs/2))
    dist = len(X) * ((0.9*f0) / (fs/2))
    peaks, _ = find_peaks(X, distance = dist)
    peaks = peaks[peaks > start]
    Cn = X[peaks[0:20]]

    # Error curve
    R = np.arange(0.001, 0.5, 0.001)
    e = []
    ramp = np.linspace(0, 1, len(Cn))
    for r in R:
        n = np.arange(1, len(Cn) + 1)
        theo = np.abs((2/(n**2*np.pi**2*r*(1 - r))*np.sin(n*np.pi*r)))
        error = np.sum(np.abs(ramp*(Cn - theo/max(theo))))
        e.append(error)

    # Top peak prominence
    N = 3
    e = e/max(e)
    e_rev = max(e) - e
    e_peaks, e_prop = find_peaks(e_rev, distance = 0.05 * len(e_rev), prominence = (None, None))
    heights = e_rev[e_peaks]
    prominences = e_prop['prominences']
    top_N_indices = np.argsort(heights)[-N:]
    top_prom = prominences[top_N_indices].argmax()
    pred = R[e_peaks[top_N_indices[top_prom]]]

    return pred

def spectral_flux(signal, frameSize, hopLength):

    frameStart = 0
    frameEnd = frameStart + frameSize
    prevFrame = np.zeros(frameSize)
    spec_flux = []

    while(frameEnd < len(signal)):
        
        currFrame = signal[frameStart:frameEnd]
        X1 = np.fft.fft(currFrame, norm='ortho')
        X0 = np.fft.fft(prevFrame, norm='ortho')
        mag_diff = abs(X1) - abs(X0)
        mag_diff = (mag_diff + abs(mag_diff))/2
        spec_flux.append(sum(mag_diff))
        
        frameStart = frameStart + hopLength
        frameEnd = frameStart + frameSize
        prevFrame = currFrame

    return np.array(spec_flux)

def calcThreshold(odf,m,lamb,alpha):

    delta = np.zeros(len(odf))
    for n in range(m, len(odf)):
        delta[n] = lamb * np.median(odf[n-m:n]) + alpha * np.mean(odf[n-m:n])
    return delta

def normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px

def rec2polar(x1, y1, x2, y2):

    if (x2 == x1):
        theta = 0
        rho = x1
    elif (y2 == y1):
        theta = 90*np.pi / 180
        rho = y1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        theta = -np.arctan(1 / m)
        rho = np.sin(theta) * b

    return rho, theta


def polar2rec(rho, theta):

    if np.sin(theta) != 0:
        x1 = 0
        y1 = int(rho/np.sin(theta))
        x2 = 10000
        y2 = int((rho - x2*np.cos(theta))/np.sin(theta))
    else:
        y1 = 0
        x1 = int(rho/np.cos(theta))
        y2 = 10000
        x2 = x1

    return x1, y1, x2, y2


def polar2recv2(rho, theta, rho1, theta1, rho2, theta2):

    b1 = (rho1-rho*(np.cos(theta-theta1)))/(np.sin(theta1-theta))
    b2 = (rho2-rho*(np.cos(theta-theta2)))/(np.sin(theta2-theta))

    x1 = int(rho*np.cos(theta)+b1*(-np.sin(theta)))
    y1 = int(rho*np.sin(theta)+b1*(np.cos(theta)))
    x2 = int(rho*np.cos(theta)+b2*(-np.sin(theta)))
    y2 = int(rho*np.sin(theta)+b2*(np.cos(theta)))

    return x1, y1, x2, y2


def tri2rect(three_points):

    pt1 = three_points[0]
    pt2 = three_points[1]
    pt3 = three_points[2]

    rho1_1, theta1 = rec2polar(pt1[0], pt1[1], pt2[0], pt2[1])
    rho1_2 = pt3[0]*np.cos(theta1) + pt3[1]*np.sin(theta1)

    theta2 = theta1 - 90*np.pi/180 if theta1 > 0 else theta1 + 90*np.pi/180
    rho2_1 = pt1[0]*np.cos(theta2) + pt1[1]*np.sin(theta2)
    rho2_2 = pt2[0]*np.cos(theta2) + pt2[1]*np.sin(theta2)

    x1, y1, x2, y2 = polar2recv2(
        rho2_1, theta2, rho1_1, theta1, rho1_2, theta1)
    x3, y3, x4, y4 = polar2recv2(
        rho2_2, theta2, rho1_1, theta1, rho1_2, theta1)
    # new_pt1 = (x1, y1)
    new_pt2 = (x2, y2)
    # new_pt3 = (x3, y3)
    new_pt4 = (x4, y4)

    return [pt1, pt2, new_pt2, new_pt4]

def extract_params(ref_point):

    pt1, pt2, pt3, pt4 = ref_point

    rho1_1, theta1 = rec2polar(pt1[0], pt1[1], pt2[0], pt2[1])
    rho1_2, _ = rec2polar(pt3[0], pt3[1], pt4[0], pt4[1])
    rho2_1, theta2 = rec2polar(pt1[0], pt1[1], pt3[0], pt3[1])
    rho2_2, _ = rec2polar(pt2[0], pt2[1], pt4[0], pt4[1])

    str_theta = theta1
    fr_theta = theta2
    upper_edge = rho1_1 if rho1_1 < rho1_2 else rho1_2
    lower_edge = rho1_2 if rho1_1 < rho1_2 else rho1_1
    left_edge = rho2_1 if rho2_1 < rho2_2 else rho2_2
    right_edge = rho2_2 if rho2_1 < rho2_2 else rho2_1
    fr_length = abs(lower_edge - upper_edge)
    str_length = abs(right_edge - left_edge)

    return str_theta, fr_theta, upper_edge, lower_edge, left_edge, right_edge, fr_length, str_length

def rotate_image(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def polar_rotated(rho, theta, angle, rot_mat):

    theta_rotated = theta - angle*np.pi/180
    x1, y1, x2, y2 = polar2rec(rho, theta)
    (x1_rotated, y1_rotated) = np.dot(rot_mat, [x1, y1, 1])
    rho_rotated = x1_rotated * np.cos(theta_rotated) + y1_rotated*np.sin(theta_rotated)

    return rho_rotated, theta_rotated


def find_near(array, value, threshold):

    array = np.asarray(array)
    boolArr = (array > (value - threshold)) & (array < (value + threshold))

    return array[boolArr]

def spectral_centroid(signal, fs, frameSize = 2048, hopLength = 441):

    frameStart = 0
    frameEnd = frameStart + frameSize
    prevFrame = signal[frameStart:frameEnd]
    spec_centroid = []

    while(frameEnd < len(signal)):
        
        currFrame = signal[frameStart:frameEnd]
        magnitudes = np.abs(np.fft.rfft(currFrame))
        length = len(currFrame)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2 + 1])
        spec_centroid.append(np.sum((magnitudes ** 2) * freqs) / np.sum(magnitudes ** 2))
            
        frameStart = frameStart + hopLength
        frameEnd = frameStart + frameSize
        prevFrame = currFrame

    return np.mean(np.array(spec_centroid))
def spectral_flatness(signal):

    magnitudes = np.abs(np.fft.rfft(signal))
    gmean = np.exp(np.mean(np.log( magnitudes ** 2), keepdims=True))
    amean = np.mean( magnitudes ** 2, keepdims=True)
    return gmean / amean

def finger_angles(finger_coordinates):

    # Fit plane
    centroid = np.mean(finger_coordinates, axis = 0)
    u, s, vh = np.linalg.svd(finger_coordinates - centroid)
    normal_plane = vh[:, 2]

    # Project points onto plane
    projected = finger_coordinates
    for n in range(len(finger_coordinates)):
        projected[n, :] = finger_coordinates[n, :] - np.dot(finger_coordinates[n, :] - finger_coordinates[0, :], normal_plane)*normal_plane

    # Define angles
    v1 = projected[1,:] - projected[0,:]
    v2 = projected[2,:] - projected[1,:]
    v3 = projected[3,:] - projected[2,:]
    v4 = projected[4,:] - projected[3,:]
    theta_1 = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    theta_2 = np.arccos(np.dot(v2, v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)))
    theta_3 = np.arccos(np.dot(v3, v4)/(np.linalg.norm(v3)*np.linalg.norm(v4)))

    return [theta_1, theta_2, theta_3]

def keypoints_to_features(hand_coordinates):

    thumb_angles = finger_angles(hand_coordinates[[0, 1, 2, 3, 4], :])
    index_angles = finger_angles(hand_coordinates[[0, 5, 6, 7, 8], :])
    middle_angles = finger_angles(hand_coordinates[[0, 9, 10, 11, 12], :])
    ring_angles = finger_angles(hand_coordinates[[0, 13, 14, 15, 16], :])
    little_angles = finger_angles(hand_coordinates[[0, 17, 18, 19, 20], :])
    features = np.array([thumb_angles, index_angles, middle_angles, ring_angles, little_angles]).flatten()

    return features

def calcFlux(currFrame, prevFrame):
    X1 = np.fft.fft(np.multiply(currFrame,np.hamming(len(currFrame))))
    X0 = np.fft.fft(np.multiply(prevFrame,np.hamming(len(prevFrame))))
    mag_diff = abs(X1) - abs(X0)
    mag_diff = (mag_diff + abs(mag_diff)) / 2
    return sum(mag_diff)

def checkOnset(odf, lamb, alpha):
    onset = False
    delta = lamb * np.median(odf[0:-2]) + alpha * np.mean(odf[0:-2])
    if odf[-2] > odf[-3] and odf[-2] > odf[-1] and odf[-2] > delta:
        onset = True
    return onset

def freq_from_autocorr(signal, fs):
    # Calculate autocorrelation (same thing as convolution, but with one input
    # reversed in time), and throw away the negative lags

    signal -= np.mean(signal)   # Remove DC Offset
    corr = fftconvolve(signal, signal[::-1], mode = 'full')
    corr = corr[len(corr)//2:]
    
    # Find the first peak on the left
    peaks = peakutils.indexes(corr, thres = 0.6, min_dist = 5)
    
    if len(peaks) > 0:
        i_peak = peakutils.indexes(corr, thres = 0.6, min_dist = 5)[0]
        i_interp = parabolic(corr, i_peak)[0]
        freq = fs / i_interp
    else:
        freq = 0
        i_interp = 0

    return freq, corr, i_interp

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x - 1] - 2*f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def tonalMap(X1, X2, Y):
    
    X3 = (X1 - np.mean(X1)) * (X2 - np.mean(X2))
    X = np.stack([X1, X2, X3], axis = 1)
    X = sm.add_constant(X)
    # Y = (4)*(Y - np.min(Y)) / (np.max(Y) - np.min(Y)) - 2

    model = sm.OLS(Y, X)
    results = model.fit()

    x1, x2, = np.meshgrid(np.linspace(0, 30, 1000), np.linspace(0, 60, 1000))
    y = results.params[0] + x1 * results.params[1] + x2 * results.params[2] + (x1 - np.mean(x1)) * (x2 - np.mean(x2)) * results.params[3]
    y = (4) * (y - y.min()) / (y.max() - y.min()) - 2
    print(results.summary())

    return x1, x2, y, results

def intersect_point_line(p0, p1, p2):

    n = (p2 - p1) / np.linalg.norm(p2 - p1)
    ap = (p0 - p1)

    t = np.dot(ap, n)
    x = p1 + t * n
    ratio = np.linalg.norm(p2 - x) / np.linalg.norm(p2 - p1)

    if ratio > 0.5:
        ratio = 1 - ratio
    
    return ratio

def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=90., elev=90.)

def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=90., elev=90.)

def finger_angles(finger_coordinates):

    # Fit plane
    centroid = np.mean(finger_coordinates, axis = 0)
    u, s, vh = np.linalg.svd(finger_coordinates - centroid)
    normal_plane = vh[:, 2]

    # Project points onto plane
    projected = finger_coordinates
    for n in range(len(finger_coordinates)):
        projected[n, :] = finger_coordinates[n, :] - np.dot(finger_coordinates[n, :] - finger_coordinates[0, :], normal_plane)*normal_plane

    # Define angles
    v1 = projected[1,:] - projected[0,:]
    v2 = projected[2,:] - projected[1,:]
    v3 = projected[3,:] - projected[2,:]
    v4 = projected[4,:] - projected[3,:]
    theta_1 = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    theta_2 = np.arccos(np.dot(v2, v3)/(np.linalg.norm(v2)*np.linalg.norm(v3)))
    theta_3 = np.arccos(np.dot(v3, v4)/(np.linalg.norm(v3)*np.linalg.norm(v4)))

    return [theta_1, theta_2, theta_3]

def keypoints_to_features(hand_coordinates):

    thumb_angles = finger_angles(hand_coordinates[[0, 1, 2, 3, 4], :])
    index_angles = finger_angles(hand_coordinates[[0, 5, 6, 7, 8], :])
    middle_angles = finger_angles(hand_coordinates[[0, 9, 10, 11, 12], :])
    ring_angles = finger_angles(hand_coordinates[[0, 13, 14, 15, 16], :])
    little_angles = finger_angles(hand_coordinates[[0, 17, 18, 19, 20], :])
    features = np.array([thumb_angles, index_angles, middle_angles, ring_angles, little_angles]).flatten()

    return features