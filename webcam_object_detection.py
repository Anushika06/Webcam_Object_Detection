# Import libraries
import cv2   
from ultralytics import YOLO
import time

# Load pre-trained YOLOv8 model (tiny version for speed)
model = YOLO("yolov8s.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(3, 640)  # frame width: 640 pixels
cap.set(4, 480)  # frame height: 480 pixels

# Initialize FPS calculation
prev_time = 0

print("Starting webcam object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    #ret: Check if frame is captured successfully (boolean)
    #frame: The captured frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    #Returns a list of detection results; [0] gets the first (and only) result for this frame

    # Annotate frame with bounding boxes
    annotated_frame = results.plot()

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #(10,30): Position of the text
    # 1: Font size
    #(0,255,0): Green color in BGR
    #2: Thickness of the text

    # Display annotated frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
