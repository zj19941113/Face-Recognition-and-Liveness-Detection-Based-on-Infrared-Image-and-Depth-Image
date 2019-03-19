# Face-Recognition-and-Liveness-Detection-Based-on-Infrared-Image-and-Depth-Image
## 准备:
### 1、Ubuntu C++ 编译dlib库 
https://blog.csdn.net/ffcjjhv/article/details/84660869
### 2、数据+模型下载
https://pan.baidu.com/s/1jIoW6BSa5nkGWNipL7sxVQ
其中包括：
 - candidate-face.zip（人脸库：包含29个正面人脸红外图）
 - allface.zip（测试人脸集：包括29个人，每人13种脸部姿态下的红外图与深度图）
 - shape_predictor_68_face_landmarks.dat（人脸68关键点检测器）
 - dlib_face_recognition_resnet_model_v1.dat（人脸识别模型）
 
![](https://github.com/zj19941113/Face-Recognition-and-Liveness-Detection-Based-on-Infrared-Image-and-Depth-Image/blob/master/img/006-1.png)

### 3、代码分析:
主要包含3个函数：
```c
/* 函数声明 */
/* 人脸库训练 */
int candidates_train(const char *facesFile,std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates);
```
运行candidates_train，遍历人脸库candidate-face文件夹，将候选人名单存入candidates，将候选人人脸特征存入candidates_descriptors。
```c
/* 输出人脸位置 返回识别结果 */
string face_location(const char *imgFile,std::vector<int>&locates, std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates);
```
运行face_location，得到要测试图片的人脸特征，计算与每个候选人人脸特征的欧式距离，得到距离最小值的编号，从而在candidates中得到识别结果。在函数运行过程中，将人脸位置信息传入locates，进行活体检测。
```c
/* 判断是否为活体 */
bool liveness_detection(const char *DeepFile,std::vector<int>&locates); 
```
运行liveness_detection，利用深度图与人脸位置信息进行活体检测，主要利用了RANSAC算法。
### 4、运行结果：
![](https://github.com/zj19941113/Face-Recognition-and-Liveness-Detection-Based-on-Infrared-Image-and-Depth-Image/blob/master/img/006-3.png)

补充：
python版的看这里 https://blog.csdn.net/ffcjjhv/article/details/84637986 

python版的在allface文件夹共375张图片上的识别精度为99.469%，出错的两张是allleft姿态，侧转角度很大。模型算法和这篇c++版是一样的，只是语言不一样。可以看出识别效果还是很好的。
this_is_who.py在test-face文件夹中的批量测试结果：

![](https://github.com/zj19941113/Face-Recognition-and-Liveness-Detection-Based-on-Infrared-Image-and-Depth-Image/blob/master/img/006-5.png)
