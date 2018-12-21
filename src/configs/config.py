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
cfgs.CLS_NUM = 2 #inlcude background
# ---------------------------------------- System_config
cfgs.NET_NAME = 'resnet50'#'resnet100'  # 'mobilenetv2' 'resnet50'
cfgs.SHOW_TRAIN_INFO_INTE = 5000
cfgs.SMRY_ITER = 100000
cfgs.DATASET_NAME = 'Prison'

# ------------------------------------------ Train config
cfgs.BN_USE = True 
cfgs.WEIGHT_DECAY = 1e-5
cfgs.MOMENTUM = 0.9
cfgs.LR = [0.01,0.001,0.0001,0.00001]
cfgs.DECAY_STEP = [150000, 300000,450000]
# -------------------------------------------- Data_preprocess_config 
cfgs.DATASET_NAME = 'Prison'  # 'ship', 'spacenet', 'pascal', 'coco'
cfgs.PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
cfgs.PIXEL_NORM = 128.0
cfgs.IMG_LIMITATE = 0
cfgs.IMG_SHORT_SIDE_LEN = 480
cfgs.IMG_MAX_LENGTH = 640