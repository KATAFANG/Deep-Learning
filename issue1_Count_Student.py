# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 11:33:57 2022

@author: Fang
"""

import cv2

# 加载级联分类器
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image_path="03R.jpg"

image = cv2.imread(image_path) # 导入原图

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 原图灰度化

# 用级联分类器识别人像
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),) 

count=0

# 标记原图并累计人数
for (x, y, width, height) in faces: 
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    count+=1

# 输出和保存
print('Number of faces in total: %s' % count)

cv2.imwrite("signed.jpg",image)

cv2.waitKey(0)
