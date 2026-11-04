import time
import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"   

# Pairs of landmark indices to connect with a line
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # closes the palm
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    # Convert each landmark from fractions to pixel coordinates
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1, pt2 = points[start_idx], points[end_idx]

        cv2.line(frame, pt1, pt2, (120, 100, 0), 5, cv2.LINE_AA)
        cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

    
    for idx, (x, y) in enumerate(points):
        
        if idx in FINGERTIP_INDICES:
            
            cv2.circle(frame, (x, y), 6, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
        
        else:
            
            cv2.circle(frame, (x, y), 4, (180, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


# 0, 9 and 10: stay silent
sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    2: "sounds_swaras/2_Re.mp3",
    3: "sounds_swaras/3_Ga.mp3",
    4: "sounds_swaras/4_Ma.mp3",
    5: "sounds_swaras/5_Pa.mp3",
    6: "sounds_swaras/6_Dha.mp3",
    7: "sounds_swaras/7_Ni.mp3",
    8: "sounds_swaras/8_Sa_upper.mp3",
}
swara_names = {
    1: "Sa", 2: "Re", 3: "Ga", 4: "Ma",
    5: "Pa", 6: "Dha", 7: "Ni", 8: "Sa (upper)",
}


pygame.mixer.init()

STABLE_FRAMES = 5     # about 0.15 s at 30 fps
last_seen = -1        # count seen in the previous frame
frames_same = 0       # frames in a row with the same count
last_played = -1      # count we last triggered on
current_swara = ""    # name shown on screen

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                          # two hands -> counts up to 10
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.time()                  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue                          # skip this frame if the read failed

    frame = cv2.flip(frame, 1)           

    # ---- Milestone 3: detect hands ----
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    count = 0
    for hand_landmarks in result.hand_landmarks:
        if hand_landmarks[5].x < hand_landmarks[17].x:
            if hand_landmarks[4].x < hand_landmarks[3].x:
                count += 1
        else:
            if hand_landmarks[4].x > hand_landmarks[3].x:
                count += 1

        for tip in [8, 12, 16, 20]:
            if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
                count += 1

        draw_landmarks(frame, hand_landmarks)

    if count == last_seen:
        frames_same += 1
    else:
        last_seen = count
        frames_same = 1

    if frames_same >= STABLE_FRAMES and count != last_played:
        last_played = count
        if count in sounds_mapping:
            pygame.mixer.music.load(sounds_mapping[count])
            pygame.mixer.music.play()     
            current_swara = swara_names[count]
        else:
            current_swara = ""            

    
    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1.3

    # 1. Finger Count 
    cv2.putText(frame, f"FINGERS ~ {count}", (25, 45), font, scale, (255, 255, 0), 5, cv2.LINE_AA)

    # 2. Swara Display 
    if current_swara:
        cv2.putText(frame, f"SWARA   ~ {current_swara.upper()}", (25, 75), font, scale, (255, 100, 255), 3, cv2.LINE_AA)

    cv2.imshow("Swara Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
