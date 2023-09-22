import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load video
video_path = "TestVideo.avi"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(frame_rate * 10)  # Capture an image every 10 seconds

count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    count += 1
    
    if count % frame_interval == 0:
        # Save the frame as an image
        image_name = f"frame_{count // frame_interval}.jpg"
        cv2.imwrite(image_name, frame)

        # Load the saved image for object detection
        img = cv2.imread(image_name)
        height, width, _ = img.shape

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set the input to the YOLO network
        net.setInput(blob)

        # Get the output layer names
        output_layers = net.getUnconnectedOutLayersNames()

        # Run YOLO object detection
        detections = net.forward(output_layers)

        people_count = 0

        # Loop through the detections and count people
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "person":
                    people_count += 1

        print(f"Image {count // frame_interval}: Number of people detected = {people_count}")

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
