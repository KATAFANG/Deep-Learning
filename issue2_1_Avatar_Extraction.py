# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:57:32 2022

@author: Fang
"""

import cv2
import os

# 设置路径
CASE_PATH = "haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = 'me/'
DATASET_DIR = 'jm/'

# 加载级联分类器
face_cascade = cv2.CascadeClassifier(CASE_PATH)

# 按一定尺寸保存图像
def save_feces(img, name,x, y, width, height):
    image = img[y:y+height, x:x+width]
    cv2.imwrite(name, image)

image_list = os.listdir(RAW_IMAGE_DIR) # 列出文件夹下所有的目录与文件

count = 1

# 按顺序截取人像并保存
for image_path in image_list:
    image = cv2.imread(RAW_IMAGE_DIR + image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(5, 5), )

    for (x, y, width, height) in faces:
        save_feces(image, '%ss%d.bmp' % (DATASET_DIR, count), x, y - 30, width, height+30)
    count += 1
