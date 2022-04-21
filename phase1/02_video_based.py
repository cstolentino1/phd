import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.signal import find_peaks

sys.path.insert(0, '../utils')
import utils

def shape_selection(event, x, y, flags, param):
    # Grab references to the global variables
    global ref_point, crop

    # If the left mouse button was clicked, draw the point
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.append((x, y))
        
        if len(ref_point) == 1:
            cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        elif len(ref_point) == 2:
            cv2.line(image, ref_point[0], ref_point[1], (0, 0, 255), 2)
        elif len(ref_point) == 3:
            fourpoints = utils.tri2rect(ref_point)
            cv2.line(image, fourpoints[0], fourpoints[1], (0, 0, 255), 2)
            cv2.line(image, fourpoints[0], fourpoints[2], (0, 0, 255), 2)
            cv2.line(image, fourpoints[1], fourpoints[3], (0, 0, 255), 2)
            cv2.line(image, fourpoints[2], fourpoints[3], (0, 0, 255), 2)
            ref_point = fourpoints

        cv2.imshow('Image', image)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

WIDTH = 1280
HEIGHT = 720

# For video:
hands = mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
        max_num_hands = 2)

# Load video
filename = '../data/video/phase1/01_02.mkv'
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract audio from the video file
video = VideoFileClip(filename)
audio = video.audio
fs = audio.fps
signal = audio.to_soundarray(nbytes=4,buffersize=1000,fps=fs)

# Convert to mono and normalize
signal = signal.sum(axis=1) / 2
signal = signal/max(abs(signal))

# Compute spectral flux per frame
frameSize = np.floor(0.10*fs).astype(int)
hopLength = np.floor(0.02*fs).astype(int)
spec_flux = utils.spectral_flux(signal, frameSize, hopLength)

# Peak finding
peaks, _ = find_peaks(spec_flux, height = 20, distance = 40)
peaks = peaks[2:]
sample_onsets = (peaks*hopLength).astype(int)
time_onsets = peaks*hopLength/fs
frame_onsets = np.ceil(time_onsets*fps).astype(int)

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

# cv2.imwrite('initial-fretboard.jpg', image)
cv2.destroyAllWindows()

# Extract parameters
init_str_theta, init_fr_theta, init_upper_edge, init_lower_edge, init_left_edge, init_right_edge, init_fr_length, init_str_length = utils.extract_params(
    ref_point[0:4])

# Extract parameters at each onset
PP = np.zeros(((len(frame_onsets))))
duration = []
elapsed = []
for n in range(len(frame_onsets)):

    video_onset = frame_onsets[n]
    cap.set(1, video_onset)

    frame_number = 0
    plucking_positions = []
    new_upper_edges = []
    new_lower_edges = []
    new_right_edges = []
    new_left_edges = []

    # plucking finger: 0 - index, 1 - middle, 2 - ring
    sample_number = n % 10
    if sample_number in [0, 4, 8]:
        pf_index = 8
    elif sample_number in [1, 3, 5, 7, 9]:
        pf_index = 12
    else:
        pf_index = 16

    start1 = time.time()
    while frame_number <= 30:
        start2 = time.time()

        success, image = cap.read()
        image = cv2.resize(image, (WIDTH, HEIGHT))
        frame_number = int(cap.get(1)) - video_onset
        if not success:
            print('Ignoring empty camera frame.')
            break

        # Hand annotation
        hand_annotations = np.copy(image) * 0
        image_hand = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_hand.flags.writeable = False
        results = hands.process(image_hand)

        # Draw the hand annotations of the image
        right_hand_index = None
        if results.multi_hand_landmarks:
            for index, handedness in enumerate(results.multi_handedness):
                if (handedness.classification[0].label) == "Right":
                    right_hand_index = index
            
            for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(hand_annotations, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check knuckle keypoints and index finger keypoint of right hand
                if index == right_hand_index:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        if idx == pf_index:
                            normalized_landmark = utils.normalized_to_pixel_coordinates(landmark.x, landmark.y, WIDTH, HEIGHT)
                            plucking_finger = (int(WIDTH-normalized_landmark[0]-1), int(normalized_landmark[1]))
        
        hand_annotations = cv2.flip(hand_annotations, 1)

        # Line detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        boolArr = (lines_plr[:,0] > init_upper_edge - 0.1*init_fr_length) & (lines_plr[:,0] < init_lower_edge + 0.1*init_fr_length)
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
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image_rotate = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags = cv2.INTER_LINEAR)
        gray_rotate = cv2.cvtColor(cv2.GaussianBlur(image_rotate, (3, 3), 0), cv2.COLOR_BGR2GRAY)

        # Horizontal Sobel Filter and Post-Processing
        grad_x = cv2.Sobel(gray_rotate, cv2.CV_16S, 
            1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        (thresh, abs_grad_x) = cv2.threshold(abs_grad_x, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        abs_grad_x = cv2.medianBlur(abs_grad_x, 3)
        abs_grad_x = cv2.dilate(abs_grad_x, np.ones((3, 3), np.uint8), iterations = 1)
        abs_grad_x = cv2.erode(abs_grad_x, np.ones((3, 3), np.uint8), iterations = 1)
        abs_grad_x = cv2.dilate(abs_grad_x, np.ones((9, 3), np.uint8), iterations = 1)

        # Apply Mask
        upper_edge_rotated, _ = utils.polar_rotated(init_upper_edge, str_theta, angle, rot_mat)
        lower_edge_rotated, _ = utils.polar_rotated(init_lower_edge, str_theta, angle, rot_mat)
        right_edge_rotated, _ = utils.polar_rotated(init_right_edge, fr_theta, angle, rot_mat)
        left_edge_rotated,  _ = utils.polar_rotated(init_left_edge, fr_theta, angle, rot_mat)
        min_y, max_y = int(upper_edge_rotated - 0.50*init_fr_length), int(lower_edge_rotated + 0.50*init_fr_length)
        min_x, max_x = int(left_edge_rotated - 0.10*init_str_length), int(right_edge_rotated + 0.10*init_str_length)

        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[min_y: max_y, min_x: max_x] = 255
        abs_grad_x_masked = cv2.bitwise_and(abs_grad_x, abs_grad_x, mask = mask)
        abs_grad_x_binary = (abs_grad_x_masked > 0).astype(int)

        # Fret Detection
        fret_candidates = []
        start_cols = []
        end_cols = []
        upper_edges = []
        lower_edges = []
        flag = 0
        for col in range(min_x, max_x -1):
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
            if (current_height > 0.5 * init_fr_length) & (current_height < 1.5 * init_fr_length):
                fret_candidates.append(col)
                upper_edges.append(upper)
                lower_edges.append(lower)
                if flag == 0:
                    start_col = col
                    flag = 1
            else:
                if flag == 1:
                    end_col = col
                    flag = 0
                    start_cols.append(start_col)
                    end_cols.append(end_col)
        if len(end_cols) < 5:
            continue

        # Boundaries
        new_upper_edge = np.median(upper_edges).astype(int)
        new_lower_edge = np.median(lower_edges).astype(int)
        new_right_edge = int(end_cols[-1])
        left_edge_candidate1 = int(new_right_edge - init_str_length)
        left_edge_candidate2 = utils.find_near(start_cols, left_edge_candidate1, 0.05*init_str_length)
        new_left_edge = max(left_edge_candidate2) if len(left_edge_candidate2) > 0 else left_edge_candidate1
        new_upper_edges.append(new_upper_edge)
        new_lower_edges.append(new_lower_edge)
        new_right_edges.append(new_right_edge)
        new_left_edges.append(new_left_edge)
        new_upper_edge = np.median(new_upper_edges).astype(int)
        new_lower_edge = np.median(new_lower_edges).astype(int)
        new_right_edge = np.median(new_right_edges).astype(int)
        new_left_edge = np.median(new_left_edges).astype(int)
        mask = np.zeros((HEIGHT, WIDTH))
        mask[new_upper_edge: new_lower_edge, new_left_edge: new_right_edge] = 1

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
        fret_mask[new_upper_edge: new_lower_edge, new_right_edge - 3: new_right_edge + 3] = 255
        fret_mask[new_upper_edge: new_lower_edge, new_left_edge - 3: new_left_edge + 3] = 255
        for index in range(len(start_cols)):
            middle_col = int((start_cols[index] + end_cols[index]) / 2)
            if middle_col < new_right_edge and middle_col > new_left_edge + 0.4 * init_str_length:
                fret_mask[new_upper_edge: new_lower_edge, middle_col - 1: middle_col + 1] = 255
        green_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        green_image[:] = (0, 255, 0)
        line_image2 = cv2.bitwise_and(green_image, green_image, mask = fret_mask)

        # Line Image
        line_image = cv2.addWeighted(line_image1, 1, line_image2, 1, 0)
        angle = 90 - str_theta*180/np.pi
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        line_image = cv2.warpAffine(line_image, rot_mat, image.shape[1::-1], flags = cv2.INTER_LINEAR)

        # Rotate Boundaries
        new_upper_edge, _ = utils.polar_rotated(new_upper_edge, 90*np.pi/180, angle, rot_mat)
        new_lower_edge, _ = utils.polar_rotated(new_lower_edge, 90*np.pi/180, angle, rot_mat)
        new_right_edge, _ = utils.polar_rotated(new_right_edge, 0, angle, rot_mat)
        new_left_edge, _ =  utils.polar_rotated(new_left_edge, 0, angle, rot_mat)

        # Extract plucking position
        plucked_string = np.linspace(new_upper_edge, new_lower_edge, 8)[5]
        x1, y1, x2, y2 = utils.polar2recv2(plucked_string, str_theta, new_left_edge, fr_theta, new_right_edge, fr_theta)
        relative_plucking_position = utils.intersect_point_line(np.array(plucking_finger), np.array([x1, y1]), np.array([x2, y2]))
        relative_plucking_position = relative_plucking_position if relative_plucking_position <= 0.5 else 1 - relative_plucking_position

        string_length = 64.0
        plucking_position = relative_plucking_position * string_length
        plucking_positions.append(plucking_position)
        
        # Final annotations
        annotations = cv2.addWeighted(hand_annotations, 1, line_image, 1, 0)
        final_image = cv2.addWeighted(image, 1, line_image, 1, 0)
        elapsed2 = time.time() - start2

        elapsed.append(elapsed2)

        cv2.imshow('Image', final_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    elapsed1 = time.time() - start1
    print('Time elapsed on estimation: ', elapsed1)

    duration.append(elapsed1)
    PP[n] = np.median(plucking_positions)
    print('Plucking point position is %.2f cm'%(PP[n]))

print('Time elapsed per frame: ', np.mean(elapsed))
print('Time elapsed per estimation: ', np.mean(duration))
np.save('02_video_based_plucking_positions.npy', PP)

# Results
label = []
for n in range(len(PP)):
    pluck_pos = 4 + (int(n / 10) % 5) * 6
    label.append(pluck_pos)

mae = np.mean(abs(PP-label))
print('Mean absolute error is ', mae, ' cm.')

plt.scatter(label, PP, color = 'r')
plt.plot(np.arange(0, 32), np.arange(0, 32), color = 'b')
plt.title('Plucking point position results')
plt.xlabel('True value of plucking point position (cm)')
plt.ylabel('Predicted value of plucking position (cm)')
plt.show()