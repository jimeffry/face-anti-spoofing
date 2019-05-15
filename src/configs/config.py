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
cfgs.CLS_NUM = 4 #inlcude background:0, mobile:1  tv:2 remote-control:3
cfgs.Beard = ['No_Beard','Mustache','Goatee','Beard']
cfgs.Hair = ['Other_color','Black','Blond','Brown','Gray']
cfgs.Head = ['normal','Bangs','Bald']
cfgs.FaceProperty = ['Male','Wearing_Hat','Wearing_Earrings','Wearing_Necklace','Wearing_Necktie',\
            'Eyeglasses','Young','Smiling','Arched_Eyebrows','Bushy_Eyebrows','Blurry']
# ---------------------------------------- System_config
cfgs.NET_NAME = 'lenet5'#'resnet100'  # 'mobilenetv2' 'resnet50' 'lenet5'
cfgs.SHOW_TRAIN_INFO_INTE = 100
cfgs.SMRY_ITER = 250
cfgs.DATASET_NAME = 'FaceAnti' #'Mobile' 'Prison' FaceAnti Fruit
cfgs.DATASET_LIST = ['Prison', 'WiderFace','Mobile','FaceAnti','Fruit'] 
cfgs.DATA_NAME = ['normal','fake','monitor','telecontroller'] 

# ------------------------------------------ Train config
cfgs.RD_MULT = 0
cfgs.MODEL_PREFIX = 'mobilenetv2-1_0'
cfgs.IMG_SIZE = [224,224]
cfgs.BN_USE = True 
cfgs.WEIGHT_DECAY = 1e-5
cfgs.MOMENTUM = 0.9
cfgs.LR = [0.01,0.001,0.0005,0.0001,0.00001]
cfgs.DECAY_STEP = [15000,25000,35000,50000]
# -------------------------------------------- Data_preprocess_config 
cfgs.PIXEL_MEAN = [127.5,127.5,127.5] #[123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
cfgs.PIXEL_NORM = 128.0
cfgs.IMG_LIMITATE = 0
cfgs.IMG_SHORT_SIDE_LEN = 224
cfgs.IMG_MAX_LENGTH = 224
# -------------------------------------------- test model
cfgs.ShowImg = 0
cfgs.mx_version = 1
cfgs.debug = 0
cfgs.display_model = 0
cfgs.batch_use = 0
cfgs.model_resave = 0
cfgs.time = 0
#-------------------------------------------- Face Detect Model
cfgs.rnet_out = 0 # if onet_out=1,face detect result will be output by Onet
cfgs.onet_out = 0 # if pnet_out=1,face detect result will be output by Pnet
cfgs.pnet_out = 0 #if time=1, print the time consuming by every net
cfgs.time = 0 #if crop_org=1, the main file--test.py will directly output the result by giving face detect model
cfgs.crop_org = 0 #if x_y=1, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,y1,x2,y2,...,x5,y5
cfgs.x_y = 0 #if x_y=0, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
cfgs.box_widen = 1 #if box_widen = 1, the boxes got by face detection model output , will be widened. used for images to build database
cfgs.img_downsample = 1 #whether to downsample img to short size is 320
cfgs.imgpad = 0 #whether to keep the original ratio to get ouput size
cfgs.show = 0 # whether to show the picture