# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/20 17:09
#project: face anti spoofing
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face anti spoofing
####################################################
from easydict import EasyDict 

cfgs = EasyDict()


#------------------------------------------ convert data to tfrecofr config
cfgs.BIN_DATA = 0 # whether read image data from binary
cfgs.CLS_NUM = 3 #inlcude background:0, mobile:1  tv:2 remote-control:3
# ---------------------------------------- System_config
cfgs.NET_NAME = 'mobilenetv2'#'resnet100'  # 'mobilenetv2' 'resnet50' 'lenet'
cfgs.SHOW_TRAIN_INFO_INTE = 20
cfgs.SMRY_ITER = 80
cfgs.DATASET_NAME = 'Fruit' #'Mobile' 'Prison' FaceAnti Fruit
cfgs.DATASET_LIST = ['Prison', 'WiderFace','Mobile','FaceAnti','Fruit'] 

# ------------------------------------------ Train config
cfgs.RD_MULT = 0
cfgs.MODEL_PREFIX = 'anti'
cfgs.IMG_SIZE = [112,112]
cfgs.BN_USE = True 
cfgs.WEIGHT_DECAY = 1e-5
cfgs.MOMENTUM = 0.9
cfgs.LR = [0.01,0.001,0.0005,0.0001,0.00001]
cfgs.DECAY_STEP = [30000, 40000,30000,40000]
# -------------------------------------------- Data_preprocess_config 
cfgs.PIXEL_MEAN = [127.5,127.5,127.5] #[123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
cfgs.PIXEL_NORM = 128.0
cfgs.IMG_LIMITATE = 0
cfgs.IMG_SHORT_SIDE_LEN = 224
cfgs.IMG_MAX_LENGTH = 224