# Reproduction of MobileNetV2 using MXNet for Face Anti Spoofing

***
## Project Descriptions
+ **created by :** lxy and shj
+ **Time:**  2018/12/10 15:09
+ **project** Face Anti Spoofing
+ **company:** 
+ **rversion:** 0.1
+ **tools:**   python 2.7
+ **modified:**
+ **description:** The codes for training and testing
***
## Requests
* tensorflow >= 1.5.0
* python >= 2.7.15
* opencv >= 3.4.0
* caffe
* imgaug
***
## Training Data
* The training datas are downloaded from internet,using the tool [BaiduDownload](https://github.com/kong36088/BaiduImageSpider)
* We have created the dataset including 4 classes (Mobilephone:1 TV:2 telectroller:3 background:0).
* The dataset lies on 192.168.0.9: /data/lxy_home/data/face_anti_data/
## Run Train and Test demo
Configuration parameters lies in Root/src/configs/config.py
1. directory
+  **data** is used to store training and testing data.
+  **log** is used to store traing logs.
+  **models** is used to store network parameters.
+  **src** is used to store training and testing codes.
2. train
+  **get image list** : running Root/src/prepare_data/run_script.sh to generate traing and testing data list.
+  **image augmentation**: running Root/src/utils/transform.py for image augmentation, applying for images and images with boxes and images with keypoints
+  **pack training images**: running Root/src/prepare_data/run.sh to pack training data.
+  **to train on packed images**: running Root/src/train/run.sh
3. test
+  **test one image**: python Root/src/test/demo.py --img-path1 test.jpg --gpu 0 --load-epoch 10 --cmd-type imgtest
+  **test a video**: python Root/src/test/demo.py --file-in test.mp4 --gpu 0 --load-epoch 10 --cmd-type videotest
+  **test on a test dataset**: python demo.py --file-in ../prepare_data/output/test.txt --out-file ./output/record.txt --base-dir .../test_imgs/ --load-epoch 25 --cmd-type filetest
4. video demo for face anti-spoofing
* run Root/src/face_test/run.sh
***
## HS Demo and Properties
## Results on Test data
class|TPR|FPR|Precision|
:---:|:---:|:---:|:---:|
Mobilephone   |0.631|0.026|0.856|
TV            |0.954|0.120|0.700|
Teleconrtoller|0.827|0.013|0.929|
background    |0.809|0.106|0.812|