from imageai.Detection import ObjectDetection
import os
import cv2 
import time

camera = cv2.VideoCapture(0)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join('D:/Codes/ImageAI/venv/yolov3.pt'))
detector.loadModel()
detector.useCPU()

detection = []

while camera.isOpened():
    ret, frame = camera.read()

    _, detection = detector.detectObjectsFromImage(input_image=frame, output_type="array")
    print(detection)

    for obj in detection:
        coord = obj['box_points']
        cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255))
        cv2.putText(frame, obj['name'], (coord[0], coord[1] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Test', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

#video_path = detector.detectObjectsFromVideo(camera_input=camera,
#                                            output_file_path=os.path.join('camera_out'),
#                                            frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
#print(video_path)


###########################

