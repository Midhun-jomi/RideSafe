import cv2
import pygame
from ultralytics import YOLO

# Initialize Pygame for sound alert
pygame.mixer.init()
alert_sound = "warning1.mp3"  # Ensure this file is in the same folder

# Load YOLOv8 Nano Model
model = YOLO("yolov8n.pt")

# Open Camera
cap = cv2.VideoCapture(1)  # 0 for default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Flag to manage sound alert loop
sound_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLOv8 Detection
    results = model(frame)

    person_detected = False

    # Initialize color to avoid undefined error
    color = (0, 255, 0)  # Default color set to green

    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            class_name = model.names[int(class_id)]

            if class_name == "person" and score > 0.88:
                person_detected = True
                color = (0, 0, 255)  # Blue for person detected
            else:
                color = (0, 255, 0)  # Green for no detection

            label = f"{class_name}: {score:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sound alert if a person is detected
    if person_detected:
        print("⚠️ Person detected! Playing sound in loop...")
        if not sound_playing:
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play(-1)  # Loop alert sound
            sound_playing = True
    else:
        if sound_playing:
            pygame.mixer.music.stop()  # Stop sound when no person detected
            sound_playing = False

    # Display the video feed
    cv2.imshow("Person Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
