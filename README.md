# COMP9517-Arcadia 
## 任务及其分工
### week6
确定自己想要写的模型，了解这些模型，写这些模型的代码并封装起来，要求的效果是，能够直接在其他函数里面调用
大家把自己想要写的模型写在下面

`Franco:`
- CNN - LeNet
- CNN - VGG
- CNN - ResNet
- CNN - Faster R-CNN

`Chenyi Li (Heloise):`
- YOLO
- Transformer - DETR
- XAI (Maybe if have time)

`Anna:`
机器学习部分
- SIFT/LBP
- SVM/KNN

 `jjanjamm:`
 YOLOv8+MobileNet

## 任务简报
`目标`： 识别农业害虫
`数据集`：
- Kaggle AgroPest-12 (https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)
- 训练集: 11,502
- 验证集: 1,095
- 测试集: 546
- 类别: 12
- 附有真实标签和边界框信息
`检测器 & 分类器`
`注明 papers, tools, repositories`
## 参考文献
前三个是老师推荐的，你们有什么推荐的可以后续加在后面
1. Sourav Chakrabarty et al. Application of artificial intelligence in insect pest identification – a  review. Artificial Intelligence in Agriculture 16(1):44-61, March 2026.  https://doi.org/10.1016/j.aiia.2025.06.005  
2. Kaili Wang et al. AP162: A large-scale dataset for agricultural pest recognition. Computers and  Electronics in Agriculture 237B:110520, October 2025.  https://doi.org/10.1016/j.compag.2025.110520  
3. Xiaoping Wu et al. IP102: A large-scale benchmark dataset for insect pest recognition.  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.  https://doi.org/10.1109/CVPR.2019.00899

## 代码结构
### Excute.ipynb 
是我们的入口文件，也相当于我们的用户界面，我们调参，调用模型，切换数据都是在这里进行的。
### DataProcess.py
将数据集封装成一个类，到时候直接根据数据路径构建新实例，内置多种数据处理函数以响应不同的模型需求
### Model 文件夹
里面包含封装好的各种模型，诸如YOLO, VGG, ResNet等
### Data 文件夹
里面放了需要处理的图片数据
## 注意事项
1. 超参数管理：超参数统一写在一起，方便调参，并且每一个参数的用途尽量写注释
2. 每一个模型或者封装的类型或者方法，可以试着写一段简介注释

## 关于模型
目前我的感觉是这样的组合
	`数据处理(数据增强\手工特征)` + `图像识别学习模型(神经网络\机器学习)` + `目标检测学习模型(YOLO)`
### 目标检测
Yolo 不同版本的比较，比如几个典型的版本 (v3, v5, v8, v11)
### 图像识别
#### 深度学习
`神经网络模型`：我这写过的几个说不定都能用
- VGG (最经典的古老模型，可以加进去凑个数) 
- GoogleNet
- DenseNet
- ResNet (算是一个小里程碑式的更新了，感觉很有必要加进去) 
- Transformer (我自己还不太熟悉，不过感觉要想拿高分可能得加上去)
`数据增强`:
- 随机翻转
- Mosaic 拼接： 将多个图像拼接成一个大图，增加目标丰富度，提升模型对小目标的检测能力
- 颜色抖动：随机调整图像的亮度，对比度，饱和度，来适应不同光照和色彩环境
- 随机裁剪：学习局部特征，应对遮挡场景
- 高斯模糊和噪声
#### 机器学习
`手工特征`
- SIFT: 提取图像特征点，适合目标识别和匹配
- HOG: 描述目标轮廓和形状
..........
`机器学习`
- 支持向量机 SVM 
- K近邻 KNN
- 随机森林 Random Forest
- 逻辑回归
