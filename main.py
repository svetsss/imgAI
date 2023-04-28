from imageai.Detection import ObjectDetection
import os
import cv2
import time

camera = cv2.VideoCapture(0)

execution_path = os.getcwd()
detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "D:/Codes/ImageAI/venv/retinanet_resnet50_fpn_coco-eeacb38b.pth"))


# detector.setModelTypeAsTinyYOLOv3()
# detector.setModelPath("D:/Codes/ImageAI/venv/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()


finish = 0

array_detection = []

while camera.isOpened():
    ret, frame = camera.read()

    start = time.time()
    if start - finish > 1:
        _, array_detection=detector.detectObjectsFromImage(input_image=frame, input_type="array", output_type="array")
        finish = time.time()
        print (array_detection)

    for obj in array_detection:
        coord = obj['box_points']
        cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255))
        cv2.putText(frame, obj['name'], (coord[0], coord[1] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Test', frame)
        if cv2.waitkey(25) & 0xFF == ord('q'):
            break

cv2.destroyAllwindows()
