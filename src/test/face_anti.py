# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/24 11:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face  
####################################################
import tensorflow as tf 
import numpy as np 
from scipy.spatial import distance
import time 
import os 
import sys
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
import mobilenetV2
import resnet
import lenet5
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_data'))
from image_preprocess import norm_data

class Face_Anti_Spoof(object):
    def __init__(self,model_file,img_size,gpu_num):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        graph = tf.Graph()
        self.h,self.w = img_size
        with graph.as_default():
            self.image_op = tf.placeholder(name="img_inputs",shape=[1, self.h,self.w, 3], dtype=tf.float32)
            self.net_out = self.get_base_net()
            #logit = arcface_loss(embedding=self.net.outputs, labels=labels, w_init=w_init_method, out_num=85164)
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth=True  
            tf_config.log_device_placement=False
            self.sess = tf.Session(config=tf_config)
            #saver = tf.train.import_meta_graph('../models/tf-models/InsightFace_iter_350000.meta')
            saver = tf.train.Saver(max_to_keep=100)
            saver.restore(self.sess,model_file)
            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            #for var_ in all_vars:
                #print(var_)
            #for v_name in tf.global_variables():
             #   print("name : ",v_name.name[:-2],v_name.shape) 

    def get_base_net(self):
        if cfgs.NET_NAME in 'mobilenetv2':
            logits = mobilenetV2.get_symble(self.image_op,class_num=cfgs.CLS_NUM,train_fg=False)
        elif cfgs.NET_NAME in ['resnet50','resnet100']:
            logits = resnet.get_symble(self.image_op,class_num=cfgs.CLS_NUM,train_fg=False)
        elif cfgs.NET_NAME in ['lenet5','lenet']:
            logits = lenet5.get_symble(self.image_op,class_num=cfgs.CLS_NUM,train_fg=False)
        softmax_out=tf.nn.softmax(logits)
        return softmax_out

    def inference(self,img):
        h_,w_,chal_ = img.shape
        #print("img shape ",img.shape)
        #img = np.expandim(img,0)
        if h_ !=self.h or w_ !=self.w:
            tf_img = cv2.resize(img,(self.w,self.h))
            #tf_img = Img_Pad(img,(self.h,self.w))
        else:
            tf_img = img
        tf_img = tf_img[:,:,::-1]
        tf_img = norm_data(tf_img)
        caffe_img = np.expand_dims(tf_img,0)
        t = time.time()
        feat = self.sess.run([self.net_out],feed_dict={self.image_op:caffe_img})
        t_end = time.time() - t
        #print("inference time:",t_end)
        class_num = np.argmax(feat)
        return np.array(feat),class_num

    def calculateL2(self,feat1,feat2,c_type='euclidean'):
        assert np.shape(feat1)==np.shape(feat2)
        len_ = np.shape(feat1)
        #print("len ",len_)
        if c_type == "cosine":
            s_d = distance.cosine(feat1,feat2)
        elif c_type == "euclidean":
            s_d = distance.euclidean(feat1,feat2,w=1./len_[-1])
        elif c_type == "correlation":
            s_d = distance.correlation(feat1,feat2)
        elif c_type == "braycurtis":
            s_d = distance.braycurtis(feat1,feat2)
        elif c_type == 'canberra':
            s_d = distance.canberra(feat1,feat2)
        elif c_type == "chebyshev":
            s_d = distance.chebyshev(feat1,feat2)
        return s_d