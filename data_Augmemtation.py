#!/usr/bin/python
# coding:utf8

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import os

# 创建新目录
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

class data_Augmentation(object):
    # 初始化
    def __init__(self,data_path):
        self.data_oldpath = data_path

    def run(self):
        # keras数据增强方法
        datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

        # 将data\CK+Images下的图片进行数据增强并保存在新的目录下
        for root, dirs, files in os.walk(self.data_oldpath):
            print(root, '\n', dirs, '\n', files, '\n')
            new_filedir = os.path.join('dataset_preprocessing\\ADD_CK+CAS_IMAGES',root.split('\\')[-1])
            mkdir(new_filedir)
            if os.path.exists('dataset_preprocessing\\ADD_CK+CAS_IMAGES\\ADD_CK+CAS_IMAGES'):
                os.rmdir('dataset_preprocessing\\ADD_CK+CAS_IMAGES\\ADD_CK+CAS_IMAGES')
            if files:
                for file in files:
                    new_file = os.path.join(root,file)
                    img = load_img(new_file)
                    x = img_to_array(img)  # 将图像转化为(3,256,256)的矩阵
                    x = x.reshape((1,) + x.shape)  # 将图像转化为(1,3,256,256)的矩阵

                    # 使用flow模块迭代生成数据增强后的图像，保存在'dataset_preprocessing\CK+Images'文件下
                    i = 0
                    for batch in datagen.flow(x, batch_size=1,
                                              save_to_dir=new_filedir, save_format='jpeg'):
                        i += 1
                        if i > 10:
                            break  # 每张图像只扩增20张


if __name__ == '__main__':
    data = data_Augmentation("dataset\\ADD_CK+CAS_IMAGES")
    data.run()