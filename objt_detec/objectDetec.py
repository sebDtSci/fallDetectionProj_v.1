import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  
cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture vidéo")
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Détection d\'objets en temps réel', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()