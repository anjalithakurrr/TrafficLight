import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("traffic.mp4")

VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# YOLOv8 doesn't have ambulance in COCO dataset
# We simulate ambulance detection by detecting a red vehicle (for demo purposes)
# In real system you would fine-tune YOLO on ambulance images

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_val = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
lane_width = frame_width // 3

current_green = "Lane 1"
frame_counter = 0
SWITCH_EVERY = 30

emergency_active = False
emergency_timer = 0
EMERGENCY_DURATION = 90  # frames to keep green wave active

def decide_green_lane(lane_counts):
    return max(lane_counts, key=lane_counts.get)

def detect_ambulance_by_color(frame):
    """
    Simulate ambulance detection using red color detection.
    In real deployment replace this with fine-tuned YOLO model.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red color range in HSV
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (180, 255, 255)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    red_pixels = cv2.countNonZero(mask)
    # If large red region detected, assume ambulance
    return red_pixels > 5000

def draw_signal(frame, lane_counts, green_lane, frame_width, frame_height, emergency):
    lanes = ["Lane 1", "Lane 2", "Lane 3"]
    for i, lane in enumerate(lanes):
        if emergency:
            # All lanes green during emergency (green wave)
            color = (0, 255, 0)
            label = "EMERGENCY GREEN"
        else:
            color = (0, 255, 0) if lane == green_lane else (0, 0, 255)
            label = "GREEN" if lane == green_lane else "RED"
        x = i * (frame_width // 3) + 10
        cv2.rectangle(frame, (x, frame_height - 60),
                      (x + 150, frame_height - 10), color, -1)
        cv2.putText(frame, f"{lane}: {label}", (x + 5, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height = frame.shape[0]
    results = model(frame, verbose=False)
    total_count = 0
    lane_counts = {"Lane 1": 0, "Lane 2": 0, "Lane 3": 0}

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in VEHICLE_CLASSES and conf > 0.4:
            total_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cx = (x1 + x2) // 2
            if cx < lane_width:
                lane_counts["Lane 1"] += 1
            elif cx < lane_width * 2:
                lane_counts["Lane 2"] += 1
            else:
                lane_counts["Lane 3"] += 1

    # Check for ambulance
    ambulance_detected = detect_ambulance_by_color(frame)
    if ambulance_detected:
        emergency_active = True
        emergency_timer = EMERGENCY_DURATION

    if emergency_timer > 0:
        emergency_timer -= 1
        emergency_active = True
    else:
        emergency_active = False

    # Decide green lane every 30 frames (only if no emergency)
    frame_counter += 1
    if frame_counter % SWITCH_EVERY == 0 and not emergency_active:
        current_green = decide_green_lane(lane_counts)

    # Draw lane dividers
    cv2.line(frame, (lane_width, 0), (lane_width, frame_height), (255, 0, 0), 2)
    cv2.line(frame, (lane_width*2, 0), (lane_width*2, frame_height), (255, 0, 0), 2)

    # Draw signals
    draw_signal(frame, lane_counts, current_green, frame_width, frame_height, emergency_active)

    # Display info
    cv2.putText(frame, f"Total: {total_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"L1:{lane_counts['Lane 1']} L2:{lane_counts['Lane 2']} L3:{lane_counts['Lane 3']}",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if emergency_active:
        # Flashing red warning banner
        cv2.rectangle(frame, (0, 110), (frame_width, 160), (0, 0, 255), -1)
        cv2.putText(frame, "EMERGENCY VEHICLE DETECTED - GREEN WAVE ACTIVE",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"GREEN: {current_green}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Traffic Light Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()