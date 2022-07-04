# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 21:43:52 2022

@author: Fang
"""

import cv2
import numpy as np
import keras
from keras.models import load_model

# 尺寸变换预处理：将较短的一侧涂黑
def resize_without_deformation(image, size = (100, 100)):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left

    image_with_border = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = [0, 0, 0])

    resized_image = cv2.resize(image_with_border, size)

    return resized_image

# 加载级联分类器模型
CASE_PATH = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(CASE_PATH)

# 加载卷积神经网络模型
face_recognition_model = keras.Sequential()
MODEL_PATH = 'face_model.h5'
face_recognition_model = load_model(MODEL_PATH)

# 使用照片，获取并灰度化
image_path="05L1.jpg"
image = cv2.imread(image_path) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 人脸检测
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
                                     minNeighbors=5, minSize=(30, 30),) 

for (x, y, width, height) in faces:
    img = image[y:y+height, x:x+width]
    img = resize_without_deformation(img)
 
    img = img.reshape((1, 100, 100, 3))
    img = np.asarray(img, dtype = np.float32)
    img /= 255.0
 
    # result = face_recognition_model.predict_classes(img)  # keras旧版本api已失效
 
    predict_x = face_recognition_model.predict(img)         # keras新版本api可用，与上一行等效
    result = np.argmax(predict_x,axis=1)
 
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if result[0] == 15:
        cv2.putText(image, 'KANO FANG', (x, y-2), font, 0.7, (0, 255, 0), 2)
        #若认为是本人的话就标注KANO FANG

cv2.imwrite("A.jpg",image)
cv2.waitKey(0)
