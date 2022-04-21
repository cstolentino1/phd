import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

sys.path.insert(0, '../utils')
import utils

# # Load the video
# filename = '../data/video/phase3/03_02.mp4'
# cap = cv2.VideoCapture(filename)

# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = 1280
# frame_height = 720
# total_frames = cap.get(7)

# # Mediapipe
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     min_detection_confidence=0.50,
#     min_tracking_confidence=0.50,
#     max_num_hands=2)

# # Obtain onsets
# # Extract audio from the video file
# video = VideoFileClip(filename)
# audio = video.audio
# fs = audio.fps
# audio_signal = audio.to_soundarray(nbytes=4,buffersize=1000,fps=fs)

# # Convert to mono and normalize
# audio_signal = audio_signal.sum(axis=1) / 2
# audio_signal = audio_signal/max(abs(audio_signal))

# # Compute spectral flux per frame
# frameSize = np.floor(0.10*fs).astype(int)
# hopLength = np.floor(0.02*fs).astype(int)
# spec_flux = utils.spectral_flux(audio_signal, frameSize, hopLength)

# # Peak finding
# peaks, _ = find_peaks(spec_flux, height = 5, distance = 40)
# plt.plot(spec_flux)
# plt.plot(peaks, spec_flux[peaks], "x")
# plt.show()

# sample_onsets = (peaks*hopLength).astype(int)[4:]
# time_onsets = (peaks*hopLength/fs)[4:]
# video_onsets = (time_onsets*fps).astype(int)

# X = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": [],
#     "11": [], "12": [], "13": [], "14": [], "15": [], "16": [], "17": [], "18": [], "19": [], "20": [], 
#     "21": [], "22": [], "23": [], "24": [], "25": [], "26": [], "27": [], "28": [], "29": [], "30": []}
# y = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": [], "10": [],
#     "11": [], "12": [], "13": [], "14": [], "15": [], "16": [], "17": [], "18": [], "19": [], "20": [], 
#     "21": [], "22": [], "23": [], "24": [], "25": [], "26": [], "27": [], "28": [], "29": [], "30": []}

# for n in range(len(time_onsets)):
#     video_onset = video_onsets[n]
#     cap.set(1, video_onset)

#     frame_number = 0

#     # fig = plt.figure()
#     # ax = plt.axes(projection = '3d')
#     # ax.set_xlim(-1, 1)
#     # ax.set_ylim(-1, 1)
#     # ax.set_zlim(-1, 1)
#     while frame_number < 30:
#         success, image = cap.read()
#         image = cv2.resize(image, (frame_width, frame_height))
#         frame_number = int(cap.get(1) - video_onset)

#         if not success:
#             print('Ignoring empty camera frame.')
#             break

#         # Draw the hand annotations of the image
#         hand_annotations = np.copy(image) * 0
#         image_hand = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         image_hand.flags.writeable = False
#         results = hands.process(image_hand)

#         right_hand_index = None
#         if results.multi_hand_landmarks:
#             for index, handedness in enumerate(results.multi_handedness):
#                 if (handedness.classification[0].label) == "Right":
#                     right_hand_index = index
            
#             for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 if index != right_hand_index:
#                     continue
#                 mp_drawing.draw_landmarks(hand_annotations, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 hand_coordinates = np.empty((21, 3))
#                 for idx, landmark in enumerate(hand_landmarks.landmark):
#                     hand_coordinates[idx] = (landmark.x, landmark.y, landmark.z)
            
#                 hand_coordinates = hand_coordinates - hand_coordinates[0]
#                 hand_coordinates = hand_coordinates/max(np.linalg.norm(hand_coordinates, axis = 1))
#                 hand_features = utils.keypoints_to_features(hand_coordinates)

#                 # utils.plot_hand_3d(hand_coordinates, ax)
#                 # plt.show()

#         if right_hand_index == None:
#             continue

#         hand_annotations = cv2.flip(hand_annotations, 1)
#         annotated_image = cv2.addWeighted(image, 1, hand_annotations, 1, 0)
#         cv2.imshow('Image', annotated_image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#         if (n % 120) < 30:
#             label = 0
#         elif (n % 120) < 60:
#             label = 1
#         elif (n % 120) < 90:
#             label = 2
#         else:
#             label = 3
    
#         X["%d"%(frame_number)].append(hand_features)
#         y["%d"%(frame_number)].append(label)

# X = np.array(X)
# y = np.array(y)

# np.save("02_X_frames.npy", X)
# np.save("02_y_frames.npy", y)

X = np.load('02_X_frames.npy', allow_pickle='TRUE').item()
y = np.load('02_y_frames.npy', allow_pickle='TRUE').item()

X_train = np.array(X["1"])
y_train = np.array(y["1"])

model = MLPClassifier(hidden_layer_sizes = (30,), activation = 'relu', 
    solver = 'lbfgs', alpha = 0.0001, max_iter = 10000, )
model.fit(X_train, y_train)

accuracy_frame = []
for frame in range(2, 31):
    X_test = np.array(X["%d"%(frame)])
    y_test = np.array(y["%d"%(frame)])

    pred = model.predict(X_test)
    accuracy = np.sum(pred == y_test)/len(pred)
    accuracy_frame.append(accuracy)

plt.stem(range(1, 30), accuracy_frame)
plt.title('Classification accuracy for the succeeding frames')
plt.xlabel('Frame number')
plt.ylabel('Classification accuracy')
plt.show()