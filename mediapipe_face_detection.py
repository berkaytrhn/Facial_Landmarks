import cv2
import mediapipe as mp
import argparse
from time import time
import my_utils as mu
import numpy as np

def detect_face(image:np.ndarray):

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        #image = cv2.imread(image_path)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(image) # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw face detections of each face.
        if not results.detections:
            return False

        for detection in results.detections:

            x_min = detection.location_data.relative_bounding_box.xmin
            y_min = detection.location_data.relative_bounding_box.ymin
            width = detection.location_data.relative_bounding_box.width
            height = detection.location_data.relative_bounding_box.height

            img_height, img_width = image.shape[:2]

            # width -> x , column
            # height -> y, row

            x_min = int(x_min*img_width)
            y_min = int(y_min*img_height)
            
            width = int(width*img_width)
            height = int(height*img_height)

            x_max = int(x_min + width)
            y_max = int(y_min + height)


            return (x_min, y_min, x_max, y_max)
            #print(x_min, y_min)
            #print(x_max, y_max)
            #print(width, height)


            #print('Nose tip:')
            #print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            
            
            
            # image crop
            # cropped_image = annotated_image[y_min:y_max, x_min:x_max]

            # relative points
            #points = mu.read_image_points("./300VW_Dataset_2015_12_14/300VW_Dataset_2015_12_14", 1, 100)
            #relative_points = mu.relative_cropped_points(points=points,crop_coordinates=(x_min, y_min, x_max, y_max))
            
            # draw on cropped
            #drawed_cropped = mu.draw_landmarks(cropped_image, relative_points)
            
            # draw landmarks on image
            # drawed_image = mu.draw_landmarks(annotated_image, points)
            
            # cv2.rectangle function format is according to x-y coordinate system instead of row,column
           
            # draw bbox on image
            # annotated_image = cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (2555, 0, 0), 3)

            

            # cv2.imwrite("drawed_cropped_image.png", drawed_cropped)
            # cv2.imwrite("drawed_image.png", drawed_image)
            # cv2.imwrite("cropped_image.png", cropped_image)
            # cv2.imwrite('annotated_image.png', annotated_image)
            #return (x_min, y_min, x_max, y_max)

        
if __name__ == "__main__":
    _start = time()
    image = cv2.cvtColor(cv2.imread("image_100.png"), cv2.COLOR_BGR2RGB)
    detect_face(image)
    print(f"Took {time()-_start} seconds...")