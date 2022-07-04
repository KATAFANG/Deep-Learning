# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:56:37 2022

@author: Fang
"""

import cv2
import numpy as np
import keras

# 尺寸变换预处理：将图片较短的一侧涂黑
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

# 读取图片
def read_image(size = None):
    data_x, data_y = [], []
    for i in range(1,139):
        try:
            im = cv2.imread('jm/s%s.bmp' % str(i))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if size is None:
                size = (100, 100)
            im = resize_without_deformation(im, size)
            data_x.append(np.asarray(im, dtype = np.int8))
            data_y.append(str(int((i-1)/11.0)))
        except IOError as e:
            print(e)
        except:
            print('Unknown Error!')

    return data_x, data_y


from keras.layers import Conv2D, MaxPooling2D    # 引进卷积和池化层
from keras.layers import Dense, Dropout, Flatten # 引入全连接层、Dropout、Flatten
from keras.optimizers import SGD                 # 引入SGD（梯度下降优化器）来使损失函数最小化，常用的优化器还有Adam

IMAGE_SIZE = 100        # 读入所有图像及标签
raw_images, raw_labels = read_image(size=(IMAGE_SIZE, IMAGE_SIZE))
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32), np.asarray(raw_labels, dtype = np.int32) 
                                                                            # 把图像转换为float类型，方便归一化

from keras.utils import np_utils  # 对字符型类别标签进行编码
one_hot_labels = np_utils.to_categorical(raw_labels) # 使用one-hot编码规则，做到所有标签的平等化

from sklearn.model_selection import  train_test_split # 划分：训练集 ：测试集 = 4 : 1
train_input, valid_input, train_output, valid_output =train_test_split(raw_images, 
                                                                       one_hot_labels,
                                                                       test_size = 0.2)

train_input /= 255.0 # 数据归一化
valid_input /= 255.0

# 接下来构建卷积神经网络的每一层
face_recognition_model = keras.Sequential()

# 添加2个卷积层，每个卷积层含32个3*3大小的卷积核
# 边缘不补充，卷积步长向右、向下都为1，后端运算使用tf，图片输入尺寸（100，100，3），激活函数relu
face_recognition_model.add(Conv2D(32, (3, 3), padding='valid',
                                  strides = (1, 1),
                                  # data_ordering = 'tf',           # keras旧版本api已失效
                                  data_format = "channels_last",    # keras新版本api可用，与上一行等效
                                  input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
                                  activation='relu'))
 
face_recognition_model.add(Conv2D(32, (3, 3), padding='valid',
                                  strides = (1, 1),
                                  # dim_ordering = 'tf',
                                  data_format = "channels_last",
                                  activation = 'relu')) # relu作为激活函数

face_recognition_model.add(MaxPooling2D(pool_size=(2, 2))) # 池化层
face_recognition_model.add(Dropout(0.2)) # Dropout层

# Flatten层，处于卷积层与Dense（全连层）之间，将图片的卷积输出压扁成一个一维向量
face_recognition_model.add(Flatten())

# 全连接层，经典的神经网络结构，512个神经元
face_recognition_model.add(Dense(512, activation = 'relu'))
face_recognition_model.add(Dropout(0.4))

# 输出层，神经元数是标签种类数，使用sigmoid激活函数，输出最终结果
face_recognition_model.add(Dense(len(one_hot_labels[0]), activation = 'sigmoid'))

# 打印神经网络结构，检查是否搭建正确
face_recognition_model.summary()

# 使用SGD作为反向传播的优化器，来使损失函数最小化，常用的优化器还有Adam
learning_rate = 0.01    # 学习率(learning_rate)是0.01
decay = 1e-6            # 学习率衰减因子(decay)用来随着迭代次数不断减小学习率，防止出现震荡
momentum = 0.8          # 冲量(momentum),不仅可以在学习率较小的时候加速学习，又可以在学习率较大的时候减速
nesterov = True         # 使用nesterov
sgd_optimizer = SGD(lr = learning_rate, decay = decay,
                    momentum = momentum, nesterov = nesterov)

# 编译模型
face_recognition_model.compile(loss = 'categorical_crossentropy',   # 损失函数使用交叉熵
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])

# 开始训练！！！
batch_size = 20 # 每批训练数据量的大小
epochs = 100    # 共训练100次
face_recognition_model.fit(train_input, train_output,
                           epochs = epochs,
                           batch_size = batch_size, 
                           shuffle = True,
                           validation_data = (valid_input, valid_output))

# ~~~漫长的训练过程~~~

# 训练完成后在测试集上评估结果并保存模型供以后加载使用
print(face_recognition_model.evaluate(valid_input, valid_output, verbose=0))
MODEL_PATH = 'face_model.h5'
face_recognition_model.save(MODEL_PATH)
