# real-time-emotion-recognition
Using keras to recognize people emotions based on CK+ and CAS-PEAL dataset.


本项目是基于CK+和CAS-PEAL数据集和keras构建VGG16深度神经网络的7种粗表情的识别项目。

优点：
1. 可以有效识别happy，digusted , suprised , netrual四大类表情
2. 实时性高，可以用于实时识别
3. 数据集量大，可有效包含西方和东方面孔，达到比较好的识别效果


缺点：
1. 只是对于粗表情进行识别，因为数据集质量问题无法达到微表情识别
2. 因为数据集质量问题有些表情例如fearful和angry的识别效果不是那么好


提升：
1. 增强数据集，数量和质量都需要增强
2. 换用更好的方法去识别，本例使用的是深度神经网络识别


使用步骤：
1. 准备好python3及tensorflow-gpu和keras,opencv等依赖库的安装
2. 下载好CK+和CAS-PEAL数据集后解压放到dataset文件夹中
3. 运行cas-peal_image_preprocessing.py将照片分类后再手动分类
3. 运行data_Augmemtation.py后产生dataset_preprocessing数据增强后的文件夹照片
4. 运行data_Processing.py将增强后的数据集加载到ADD_CK+CAS_IMAGES48X48.pkl中并分好了训练集和测试集
5. 运行Train.py进行训练保存的模型在ADD_CK+CAS_Image_model.h5中
6. 运行Test.py进行实时测试识别你的表情
