import cv2
import numpy as np

video_path = r"C:\Users\Avantika\Downloads\fire smoke.mp4"
cap = cv2.VideoCapture(video_path)

def detect_fire(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    contours = detect_fire(frame)
    
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  

                cv2.putText(frame, "Fire and smoke Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow("Fire and Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
