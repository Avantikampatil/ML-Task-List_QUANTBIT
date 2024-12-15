import cv2
import numpy as np

config_path = "C:\\Users\\Avantika\\Downloads\\yolov3 (1).cfg"  
weights_path = "C:\\Users\\Avantika\\Downloads\\yolov3.weights"  
coco_names_path = "C:\\Users\\Avantika\\Downloads\\coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open(coco_names_path, 'r') as f:
    classes = f.read().strip().split('\n')

cap = cv2.VideoCapture(0)  

max_frames = 10  
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)  

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imwrite(f"processed_frame_{frame_count}.jpg", frame)
    print(f"Processed frame {frame_count} saved!")

    
    cv2.imshow('Traffic Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

    if frame_count >= max_frames:
        break

cap.release()
cv2.destroyAllWindows()
