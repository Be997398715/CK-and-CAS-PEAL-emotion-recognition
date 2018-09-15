#!/usr/bin/python
# coding:utf8

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys
import os
import cv2

face_expression_labels = ['OPEN_MOUTH','FROWN','CLOSE EYES','SMILE','SUPRISE']

# 创建新目录
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

class cas_peal(object):
    # 初始化
    def __init__(self,data_txt_path,data_image_path):
        self.data_image_path = data_image_path
        self.data_txt_path = data_txt_path

    def run(self):
        with open(self.data_txt_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                line_path = line.split(' ')[0]
                line_label = line_path[24:25]
                if line_label == 'O':
                    mkdir('dataset\\cas-peal\\OPEN_MOUTH')
                    img = cv2.imread(self.data_image_path+'\\'+line_path.split('	')[0]+'.tif')
                    cv2.imwrite('dataset\\cas-peal\\OPEN_MOUTH'+'\\'+line_path.split('	')[0]+'.jpg', img)
                if line_label == 'F':
                    mkdir('dataset\\cas-peal\\FROWN')
                    img = cv2.imread(self.data_image_path+'\\'+line_path.split('	')[0]+'.tif')
                    cv2.imwrite('dataset\\cas-peal\\FROWN'+'\\'+line_path.split('	')[0]+'.jpg', img)
                if line_label == 'C':
                    mkdir('dataset\\cas-peal\\CLOSE_EYES')
                    img = cv2.imread(self.data_image_path+'\\'+line_path.split('	')[0]+'.tif')
                    cv2.imwrite('dataset\\cas-peal\\CLOSE_EYES'+'\\'+line_path.split('	')[0]+'.jpg', img)
                if line_label == 'S':
                    mkdir('dataset\\cas-peal\\SUPRISE')
                    img = cv2.imread(self.data_image_path+'\\'+line_path.split('	')[0]+'.tif')
                    cv2.imwrite('dataset\\cas-peal\\SUPRISE'+'\\'+line_path.split('	')[0]+'.jpg', img)
                if line_label == 'L':
                    mkdir('dataset\\cas-peal\\SMILE')
                    img = cv2.imread(self.data_image_path+'\\'+line_path.split('	')[0]+'.tif')
                    cv2.imwrite('dataset\\cas-peal\\SMILE'+'\\'+line_path.split('	')[0]+'.jpg', img)


        # # 将data\CK+Images下的图片进行数据增强并保存在新的目录下
        # for root, dirs, files in os.walk(self.data_image_path):
        #     print(root, '\n', dirs, '\n', files, '\n')
        #     new_filedir = os.path.join('dataset_preprocessing\CK+Images',root.split('\\')[-1])
        #     mkdir(new_filedir)
        #     if os.path.exists('dataset_preprocessing\CK+Images\CK+Images'):
        #         os.rmdir('dataset_preprocessing\CK+Images\CK+Images')
        #     if files:
        #         for file in files:
        #             new_file = os.path.join(root,file)
        #             img = load_img(new_file)
        #             x = img_to_array(img)  # 将图像转化为(3,256,256)的矩阵
        #             x = x.reshape((1,) + x.shape)  # 将图像转化为(1,3,256,256)的矩阵

        #             # 使用flow模块迭代生成数据增强后的图像，保存在'dataset_preprocessing\CK+Images'文件下
        #             i = 0
        #             for batch in datagen.flow(x, batch_size=1,
        #                                       save_to_dir=new_filedir, save_format='jpeg'):
        #                 i += 1
        #                 if i > 20:
        #                     break  # 每张图像只扩增20张


if __name__ == '__main__':
    data = cas_peal("dataset\\CAS-PEAL_Images\\FaceFP_2.txt","dataset\\CAS-PEAL_Images\\cas-peal_emotion")
    data.run()        