from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8s-pose.pt', task="pose")  # pretrained YOLOv8s model with pose estimation

# Define the video source
cap = cv2.VideoCapture(0)  # Use the webcam. Replace '0' with a video file path if needed

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
four_seconds_frames = 4 * fps  # Number of frames to cover 4 seconds

# Initialize frame buffer
frame_buffer = []

# Process video
fall_detected = False
fall_frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame with the model
    result = model(frame)
    img = result.orig_img

    # Analyze results
    boxes = result.boxes  # Boxes object for bbox outputs
    for box in boxes:
        x, y, w, h = box.xywh[0]
        kpts = result.keypoints
        nk = kpts.shape[1]
        for i in range(nk):
            keypoint = kpts.xy[0, i]
            x, y = int(keypoint[0].item()), int(keypoint[1].item())
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw keypoints
        
        if w/h > 1.4:
            if not fall_detected:
                fall_detected = True
                fall_frame_count = four_seconds_frames
            cv2.putText(img, "Fallen", (int(x), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Stable", (int(x), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Manage the frame buffer and save frames if a fall was detected
    frame_buffer.append(img)
    if fall_detected:
        if fall_frame_count > 0:
            for buffered_img in frame_buffer:
                cv2.imwrite(f"frames/fall_frame_{four_seconds_frames - fall_frame_count}.jpg", buffered_img)
                fall_frame_count -= 1
            frame_buffer = []  # Clear the buffer once saved

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
