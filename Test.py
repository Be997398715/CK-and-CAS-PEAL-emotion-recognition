#!/usr/bin/python
# coding:utf8

import numpy as np
from keras.models import load_model
import cv2
import os
from PIL import Image

# 加载opencv的人脸检测器
cascPath = 'haarcascade_files\\haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(cascPath)

# 定义表情类型
num_classes = 7
class_name_1 = ['angry', 'disgusted', 'fearful', 'happy', 'netrual', 'sadness', 'surprised']
# class_name_2 = ['angry', 'disgusted', 'fearful', 'happy', 'sadness', 'surprised', 'netrual']

# 恢复训练好的模型
model_1 = load_model('ADD_CK+CAS_Image_model.h5')
# model_2 = load_model('CK+model.h5')


def predict_emotion(face_image_gray): 
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    cv2.imshow("Gray",face_image_gray)
    image = resized_img.reshape(-1, 48, 48, 1)
    # 对输入图像进行预测
    prediction_scores_1 = model_1.predict(image)
    # prediction_scores_2 = model_2.predict(image)
    name_string_1 = ''
    # name_string_2 = ''
    if np.max(prediction_scores_1)>0.5:
        max_score = np.where(prediction_scores_1==np.max(prediction_scores_1))
        name_string_1 = class_name_1[int(max_score[1])]
    # if np.max(prediction_scores_2)>0.5:
    #     max_score = np.where(prediction_scores_2==np.max(prediction_scores_2))
    #     name_string_2 = class_name_2[int(max_score[1])]
    return name_string_1
    # return name_string_1, name_string_2



# 创建一个 VideoCapture 对象
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
flag = 1        #设置一个标志，用来输出视频信息

"""创建图片保存目录"""
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

file_path = "video_test"#保存位置
mkdir(file_path)
picture_index = 0   #保存照片的索引

while(cap.isOpened()):  # 循环读取每一帧
    ret_flag , frame = cap.read()
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 人脸探测
    faces = face_classifier.detectMultiScale(
            img_gray,
            scaleFactor=1.3,
            minNeighbors=1,  # minNeighbors=5比较难检测
            minSize=(100, 100)
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if len(faces) > 0:
        max_area_face = faces[0]
        for face in faces:
            if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                max_area_face = face
        face = max_area_face
        face_image_gray = img_gray[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
        predict_emotion_1_is = predict_emotion(face_image_gray)
        if predict_emotion_1_is:
            cv2.putText(frame, predict_emotion_1_is, (x+5, y+25),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
        # predict_emotion_1_is, predict_emotion_2_is = predict_emotion(face_image_gray)
        # if predict_emotion_1_is and predict_emotion_2_is:
        #     cv2.putText(frame, predict_emotion_1_is, (x+5, y+25),
        #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
        #     cv2.putText(frame, predict_emotion_2_is, (x+5, y+55),
        #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Video_Test",frame)
    k = cv2.waitKey(1) & 0xFF 
    if  k == ord('s'): 
        fn = file_path+'\%d.jpg' % (picture_index)
        cv2.imwrite(fn, frame)
        picture_index += 1

    elif k == ord('q'): #若检测到按键 ‘q’，退出
        break