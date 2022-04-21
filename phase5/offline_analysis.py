import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import csv
import sys
from scipy.io.wavfile import write
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip

sys.path.insert(0, '../utils')
import utils

def shape_selection(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_point, crop

    # If the left mouse button was clicked, draw the point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.append((x, y))
        cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

        if len(ref_point) == 2:
            cv2.line(image, ref_point[0], ref_point[1], (0, 0, 255), 2)

        if len(ref_point) == 3:
            fourpoints = utils.tri2rect(ref_point)
            cv2.line(image, fourpoints[0], fourpoints[1], (0, 0, 255), 2)
            cv2.line(image, fourpoints[0], fourpoints[2], (0, 0, 255), 2)
            cv2.line(image, fourpoints[1], fourpoints[3], (0, 0, 255), 2)
            cv2.line(image, fourpoints[2], fourpoints[3], (0, 0, 255), 2)
            ref_point = fourpoints

        cv2.imshow('Image', image)

WIDTH = 1280
HEIGHT = 720

# Load the video
filename = '../data/video/phase5/05_06.mp4'
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)

# Load the audio
video = VideoFileClip(filename)
audio = video.audio
fs = audio.fps
audio_signal = audio.to_soundarray(nbytes=4, buffersize=1000, fps=fs)
audio_signal = audio_signal.sum(axis=1) / 2
audio_signal = audio_signal/max(abs(audio_signal))

# Compute spectral flux per frame
frameSize = np.floor(0.10*fs).astype(int)
hopLength = np.floor(0.02*fs).astype(int)
spec_flux = utils.spectral_flux(audio_signal, frameSize, hopLength)

# Peak finding
peaks, _ = find_peaks(spec_flux, height = 20, distance = 40)
sample_onsets = (peaks*hopLength).astype(int)
time_onsets = peaks*hopLength/fs
frame_onsets = np.ceil(time_onsets*fps).astype(int)
audio_onsets = (time_onsets*fs).astype(int)
video_onsets = (time_onsets*fps).astype(int)

# Initialize the list of reference points
ref_point = []
crop = False

# Get the first frame
ret, image = cap.read()
image = cv2.resize(image, (WIDTH, HEIGHT))
clone = image.copy()
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', shape_selection)

# Keep looping until 'q' key is pressed
while True:
    # Display the image and wait for a keypress
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF

    # Press 'r' to reset the window
    if key == ord('r'):
        image = clone.copy()
        ref_point = []

    # If the 'c' key is pressed, break from the loop
    elif key == ord('c'):
        break

cv2.destroyAllWindows()

# Extract parameters
init_str_theta, init_fr_theta, init_upper_edge, init_lower_edge, init_left_edge, init_right_edge, init_fr_length, init_str_length = utils.extract_params(
    ref_point[0:4])

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
    max_num_hands=2)

time_onsets = np.load('../data/csv/phase5/05_06_time_onsets.npy')
audio_onsets = (time_onsets*fs).astype(int)
video_onsets = (time_onsets*fps).astype(int)

# Extract parameters at each onset
SC = np.zeros(len(time_onsets))
PP = np.zeros(len(time_onsets))
AA = np.zeros(len(time_onsets))
PE = np.zeros(len(time_onsets))

lower_hsv1 = np.array([0, 12, 148], np.uint8)
upper_hsv1 = np.array([21, 41, 179], np.uint8)
lower_hsv2 = np.array([0, 40, 93], np.uint8)
upper_hsv2 = np.array([11, 64, 120], np.uint8)

pitch_collection = np.hstack((98.00*2**np.arange(11), 123.47*2**np.arange(11), 82.41*2**np.arange(11)))
pitch_candidates = [196, 247, 330]

# Start loop
for n in range(len(time_onsets)):

    # Audio signal analysis
    audio_onset = audio_onsets[n]
    signal = audio_signal[audio_onset:audio_onset+int(1.0*fs)]
    write('../data/audio/phase5/06/05_06_{0:03}.wav'.format(n+1), fs, signal)
    freq, corr, ip = utils.freq_from_autocorr(signal, fs)
    pitch_diff = np.abs(pitch_collection - freq)
    idx = pitch_diff.argmin()
    if pitch_diff[idx] < 20:
        pitch = pitch_candidates[int(idx / 11)]
    else:
        pitch = 0
    
    PE[n] = pitch
    SC[n] = utils.spectral_centroid(signal, fs)

    # Video signal analysis
    video_onset = video_onsets[n]
    cap.set(1, video_onset)

    frame_number = 0
    plucking_positions = []
    angles_of_attack = []
    new_upper_edges = []
    new_lower_edges = []
    new_right_edges = []
    new_left_edges = []
    while frame_number <= 5:
        success, image = cap.read()
        frame_number = int(cap.get(1)) - video_onset

        if not success:
            print('Ignoring empty camera frame.')
            break

        # Resize the frame
        cv_img = cv2.resize(image, (WIDTH, HEIGHT))
        line_image = np.copy(cv_img) * 0
        hand_annotations = np.copy(cv_img) * 0

        # Hand annotation
        image_hand = cv2.cvtColor(cv2.flip(cv_img, 1), cv2.COLOR_BGR2RGB)
        image_hand.flags.writeable = False
        results = hands.process(image_hand)
        right_hand_index = None
        if results.multi_hand_landmarks:
            for index, handedness in enumerate(results.multi_handedness):
                if (handedness.classification[0].label) == "Right":
                    right_hand_index = index

            fingers_tip = []
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(hand_annotations, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if index != right_hand_index:
                    continue
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx == 5:
                        normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                        index_knuckle = (int(WIDTH - normalized_landmark[0] - 1), int(normalized_landmark[1]))
                    elif idx == 17:
                        normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                        little_knuckle = (int(WIDTH-normalized_landmark[0]-1), int(normalized_landmark[1]))
                    elif idx == 8:
                        normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                        index_tip = (int(WIDTH-normalized_landmark[0]-1), int(normalized_landmark[1]))
                        fingers_tip.append(index_tip)
                    elif idx == 12:
                        normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                        middle_tip = (int(WIDTH-normalized_landmark[0]-1), int(normalized_landmark[1]))
                        fingers_tip.append(middle_tip)
                    elif idx == 16:
                        normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                        ring_tip = (int(WIDTH-normalized_landmark[0]-1), int(normalized_landmark[1]))
                        fingers_tip.append(ring_tip)

        if right_hand_index == None:
            continue
            
        fingers_tip = np.mean(fingers_tip, axis = 0)
        hand_annotations = cv2.flip(hand_annotations, 1)

        # Color detection
        image_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(image_hsv, lower_hsv1, upper_hsv1)
        mask_color1 = cv2.bitwise_and(cv_img, cv_img, mask = mask_hsv1)
        mask_color1 = cv2.cvtColor(mask_color1, cv2.COLOR_BGR2GRAY)
        (thresh, mask_color1) = cv2.threshold(mask_color1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_color1 = cv2.medianBlur(mask_color1, 3)
        mask_color1 = cv2.dilate(mask_color1, np.ones((3, 3), np.uint8), iterations = 1)
        mask_color1 = cv2.erode(mask_color1, np.ones((3, 3), np.uint8), iterations = 1)

        mask_hsv2 = cv2.inRange(image_hsv, lower_hsv2, upper_hsv2)
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
            threshold = 100, minLineLength = 100, maxLineGap = 20)
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
        boolArr = (lines_plr[:,0] > init_upper_edge - 0.5 * init_fr_length) & (lines_plr[:,0] < init_lower_edge + 0.5*init_fr_length)
        lines_xy = lines_xy[boolArr]
        lines_plr = lines_plr[boolArr]
        
        # Detect lines from within 10 degrees of initial string theta
        boolArr = (lines_plr[:,1] > (init_str_theta - 10*np.pi/180)) & (lines_plr[:,1] < (init_str_theta + 10*np.pi/180))
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
        upper_edge_rotated, _ = utils.polar_rotated(init_upper_edge, str_theta, angle, rot_mat)
        lower_edge_rotated, _ = utils.polar_rotated(init_lower_edge, str_theta, angle, rot_mat)
        right_edge_rotated, _ = utils.polar_rotated(init_right_edge, fr_theta, angle, rot_mat)
        left_edge_rotated, _ = utils.polar_rotated(init_left_edge, fr_theta, angle, rot_mat)
        min_y, max_y = max(int(upper_edge_rotated - 0.5 * init_fr_length), 0), min(int(lower_edge_rotated + 0.5 * init_fr_length), HEIGHT)
        min_x, max_x = max(int(left_edge_rotated - 0.20 * init_str_length), 0), min(int(right_edge_rotated + 0.20 * init_str_length), WIDTH)
        
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
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
            if (current_height > 0.50 * init_fr_length):
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
        left_edge_candidate2 = int(new_right_edge - init_str_length)
        new_left_edge = left_edge_candidate1 if left_edge_candidate1 - 0.10 * init_str_length < left_edge_candidate2 < left_edge_candidate1 + 0.10 * init_str_length else left_edge_candidate2
        # left_edge_candidate1 = int(new_right_edge - init_str_length)
        # left_edge_candidate2 = utils.find_near(start_cols, left_edge_candidate1, 0.05 * init_str_length)
        # new_left_edge = max(left_edge_candidate2) if len(left_edge_candidate2) > 0 else left_edge_candidate1
        new_upper_edges.append(new_upper_edge)
        new_lower_edges.append(new_lower_edge)
        new_right_edges.append(new_right_edge)
        new_left_edges.append(new_left_edge)
        new_upper_edge = np.median(new_upper_edges).astype(int)
        new_lower_edge = np.median(new_lower_edges).astype(int)
        new_right_edge = np.median(new_right_edges).astype(int)
        new_left_edge = np.median(new_left_edges).astype(int)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[new_upper_edge: new_lower_edge, new_left_edge: new_right_edge] = 255

        # String Mask
        strings = np.linspace(new_upper_edge, new_lower_edge, 8)[1:-1].astype(int)
        string_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        for string in strings:
            string_mask[string - 1: string + 1, new_left_edge: new_right_edge] = 255
        red_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        red_image[:] = (0, 0, 255)
        line_image1 = cv2.bitwise_and(red_image, red_image, mask = string_mask)
        
        # Fret Mask
        fret_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        fret_mask[new_upper_edge: new_lower_edge, new_right_edge - 3: new_right_edge + 3] = 1
        fret_mask[new_upper_edge: new_lower_edge, new_left_edge - 3: new_left_edge + 3] = 1
        # for index in range(len(start_cols)):
        #     middle_col = int((start_cols[index] + end_cols[index]) / 2)
        #     if middle_col < new_right_edge and middle_col > new_left_edge + 0.4 * self.init_str_length:
        #         fret_mask[new_upper_edge: new_lower_edge, middle_col - 1: middle_col + 1] = 255
        green_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        green_image[:] = (0, 255, 0)
        line_image2 = cv2.bitwise_and(green_image, green_image, mask = fret_mask)

        # Line Image
        line_image = cv2.addWeighted(line_image1, 1, line_image2, 1, 0)
        angle = 90 - str_theta * 180/np.pi
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        line_image = cv2.warpAffine(line_image, rot_mat, cv_img.shape[1::-1], flags = cv2.INTER_LINEAR)

        # Rotate Boundaries
        new_upper_edge, _ = utils.polar_rotated(new_upper_edge, 90*np.pi/180, angle, rot_mat)
        new_lower_edge, _ = utils.polar_rotated(new_lower_edge, 90*np.pi/180, angle, rot_mat)
        new_right_edge, _ = utils.polar_rotated(new_right_edge, 0, angle, rot_mat)
        new_left_edge, _ =  utils.polar_rotated(new_left_edge, 0, angle, rot_mat)

        # # Extract plucking position
        second_string = np.linspace(new_upper_edge, new_lower_edge, 8)[4]
        x1, y1, x2, y2 = utils.polar2recv2(second_string, str_theta, new_left_edge, fr_theta, new_right_edge, fr_theta)
        relative_plucking_position = utils.intersect_point_line(np.array(fingers_tip), np.array([x1, y1]), np.array([x2, y2]))
        plucking_position = relative_plucking_position * 64.0
        plucking_positions.append(plucking_position)

        # Extract angle of attack
        _, angle_knuckle = utils.rec2polar(index_knuckle[0], index_knuckle[1], little_knuckle[0], little_knuckle[1])
        angle_of_attack = (str_theta - angle_knuckle) * 180/np.pi
        angles_of_attack.append(angle_of_attack)

        # Final annotations
        annotations = cv2.addWeighted(line_image, 1, hand_annotations, 1, 0)
        updated_frame = cv2.addWeighted(cv_img, 1, annotations, 1, 0)
        cv2.imshow('image', updated_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    PP[n] = np.median(plucking_positions)
    AA[n] = np.median(angles_of_attack)

    print('PP: %.4f'%(PP[n]))
    print('AA: %.4f'%(AA[n]))

data = zip(PP, AA, SC, PE)
with open('../data/csv/phase5/05_06_data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(['plucking positions', 'angles of attack', 'spectral centroid', 'pitch estimates'])

    # write multiple rows
    writer.writerows(data)