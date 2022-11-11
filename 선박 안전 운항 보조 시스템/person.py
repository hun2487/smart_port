import torch
import numpy as np
import cv2

import math
import cv2
import mediapipe as mp
import numpy as np

import datetime

import requests
import json

import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import base64
from io import BytesIO

import winsound

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

url = "http://3.35.222.169:5000/test2" 

headers = {
    "Content-Type": "application/json" #json타입 헤더
}

engine = create_engine('mysql+pymysql://admin:admin@3.35.222.169:3306/smart_port', echo=False) #mysql
img_df = pd.read_sql(sql='select * from w_log', con=engine)
buffer = BytesIO()

degree = 0

cap = cv2.VideoCapture(1)
def rotation_matrix_to_angles(rotation_matrix):
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                        rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([x, y, z]) * 180. / math.pi

class ObjectDetection:
       
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cpu' 
        print("\n\nDevice Used:",self.device)
    

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5','yolov5s', pretrained=True)
       
        return model


    def score_frame(self, image):
        self.model.to(self.device)
        image = [image]
        results = self.model(image)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        return self.classes[int(x)]


    def plot_boxes(self, results, image):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]
        count = 0
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                if self.class_to_label(labels[i]) == 'person':
                    count += 1
                    cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)   
                    cv2.putText(image, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        cv2.putText(image, f'person: {count}', (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        if count == 0 : 
                value = "근무지 이탈!!!!"
                temp = {
                    'p_count' : 0,
                    'degree': 0
                    }
                data = json.dumps(temp)
                response = requests.post(url, headers=headers, data=data)
                print('response:',response)

        #REST API
        temp = {
            'p_count' : count,
            'degree': degree
        }
        data = json.dumps(temp)
        response = requests.post(url, headers=headers, data=data)
        print('response',response)

        b = response.json()
        if b.get('result') == False:
            winsound.Beep(
            frequency=1000,  # Hz
            duration=100  # milliseconds
            )
            cv2.imwrite('C:/Users/admin/Desktop/project/선박 안전 운항 보조 시스템/p_cap/capture.jpg', image)
            im = Image.open('C:/Users/admin/Desktop/project/선박 안전 운항 보조 시스템/p_cap/capture.jpg')
            im.save(buffer, format='jpeg')
            img_str = base64.b64encode(buffer.getvalue())
            #print(img_str)
            img_df = pd.DataFrame({'person_image':[img_str],'time':[datetime.now()]})
            img_df.to_sql('w_log', con=engine, if_exists='append',index=False)

        return image


    def __call__(self):

        while cap.isOpened():
            success, image = cap.read()

            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            f_results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            face_coordination_in_real_world = np.array([
                                                        [285, 528, 200],
                                                        [285, 371, 152],
                                                        [197, 574, 128],
                                                        [173, 425, 108],
                                                        [360, 574, 128],
                                                        [391, 425, 108]], dtype=np.float64)
            
            h, w, _ = image.shape
            face_coordination_in_image = []

            if not success:
                break
            results = self.score_frame(image)
            image = self.plot_boxes(results, image)
            
            
            if f_results.multi_face_landmarks:
                for face_landmarks in f_results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [1, 9, 57, 130, 287, 359]:
                            x, y = int(lm.x * w), int(lm.y * h)
                            face_coordination_in_image.append([x, y])

                    face_coordination_in_image = np.array(face_coordination_in_image,
                                                        dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * w
                    cam_matrix = np.array([[focal_length, 0,  w / 2],
                                        [0, focal_length, h / 2],
                                        [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Use solvePnP function to get rotation vector
                    success, rotation_vec, transition_vec = cv2.solvePnP(
                        face_coordination_in_real_world, face_coordination_in_image,
                        cam_matrix, dist_matrix)

                    # Use Rodrigues function to convert rotation vector to matrix
                    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)

                    result = rotation_matrix_to_angles(rotation_matrix)
                    
                    for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                        k, v = info
                        text = f'{k}: {int(v)}'
                        # pitch 값만 받아서 -10미만이면 print
                        if k=='pitch':
                            global degree
                            #pitch_r = f'pitch is {int(v)}'
                            degree = int(v)
                            #print(pitch_r)
                            
                        cv2.putText(image, text, (20, i*30 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
               
                 
            cv2.imshow("img", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

detection = ObjectDetection()
detection()