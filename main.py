import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("traffic.mp4")

# Class labels from COCO dataset
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)  # verbose=False stops console spam
    count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in VEHICLE_CLASSES and conf > 0.4:  # confidence filter
            count += 1

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Vehicle count overlay
    cv2.putText(frame, f"Vehicles: {count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Traffic Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
