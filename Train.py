#!/usr/bin/python
# coding:utf8

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import time, pickle
from keras.utils import to_categorical
import pickle
import os 

num_classes = 7
class_name = ['angry', 'disgusted', 'fearful', 'happy', 'netrual', 'sadness', 'surprised']


# 载入训练集与测试集
if os.path.exists('dataset_preprocessing\\ADD_CK+CAS_IMAGES48X48.pkl'):
	(X_train, y_train), (X_test, y_test) = pickle.load(open('dataset_preprocessing\\ADD_CK+CAS_IMAGES48X48.pkl',"rb"))
	print('INFO---------load pickle successfully!---------\n')
else:
	print('INFO---------load Error! Please check your pickle data!---------\n')

# 归一化并改成one-hot格式
X_train = X_train/255.           
X_test = X_test/255.
print(X_train.shape)
print(X_test.shape)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# 仿VGG16模型网络结构
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#显示卷积层各个参数
model.summary()

# rmsprop = RMSprop(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])




print('INFO---------Training---------\n')
nb_epoch = 100
batch_size = 64
start = time.time()
history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, y_test), shuffle=True)
end = time.time()
print('INFO---------@Total Time Spent: %.2f(s)---------\n' % (end - start))

# 显示训练的总曲线
def plot_acc_loss(h, nb_epoch):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
  
plot_acc_loss(history, nb_epoch)




print('INFO---------Testing ---------\n')
# 显示训练准确度和测试准确度等
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print("INFO---------Training Accuracy = %.2f %%     loss = %f---------\n" % (accuracy * 100, loss))
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("INFO---------Testing Accuracy = %.2f %%    loss = %f---------\n" % (accuracy * 100, loss))

# 保存训练好的模型
model.save('ADD_CK+CAS_Image_model.h5')
del model  #保存好模型后删除model


# x = Input(shape=(32, 32, 3))
# y = x
# y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

# y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

# y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
# y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

# y = Flatten()(y)
# y = Dropout(0.5)(y)
# y = Dense(units=nb_classes, activation='softmax', kernel_initializer='he_normal')(y)
# model1 = Model(inputs=x, outputs=y, name='model1')
# model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])