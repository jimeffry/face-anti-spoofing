# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import argparse
import string
import os 
import sys
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
import transform

def parms():
    parser = argparse.ArgumentParser(description='dataset convert')
    parser.add_argument('--VOC-dir',dest='VOC_dir',type=str,default='../../data/',\
                        help='dataset root')
    parser.add_argument('--xml-dir',dest='xml_dir',type=str,default='VOC_XML',\
                        help='xml files dir')
    parser.add_argument('--image-dir',dest='image_dir',type=str,default='VOC_JPG',\
                        help='images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='../../data/',\
                        help='tfrecord save dir')
    parser.add_argument('--save-name',dest='save_name',type=str,\
                        default='train',help='image for train or test')
    parser.add_argument('--img-format',dest='img_format',type=str,\
                        default='.jpg',help='image format')
    parser.add_argument('--dataset-name',dest='dataset_name',type=str,default='VOC',\
                        help='datasetname')
    #for widerface
    parser.add_argument('--anno-file',dest='anno_file',type=str,\
                        default='../../data/wider_gt.txt',help='annotation files')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='datasetname')
    return parser.parse_args()

class DataToRecord(object):
    def __init__(self,save_path):
        self.writer = tf.python_io.TFRecordWriter(path=save_path)

    def _int64_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
        """Wrapper for insert float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def write_recore(self,img_dict):
        # maybe do not need encode() in linux
        img_name = img_dict['img_name']
        img_height,img_width = img_dict['img_shape']
        img = img_dict['img_data']
        if not cfgs.BIN_DATA:
            img_raw = cv2.imencode('.jpg', img)[1]
            img = img_raw
        gtbox_label = img_dict['gt']
        #num_objects = img_dict['num_objects']
        feature = tf.train.Features(feature={
            'img_name': self._bytes_feature(img_name),
            'img_height': self._int64_feature(img_height),
            'img_width': self._int64_feature(img_width),
            'img': self._bytes_feature(img.tostring()),
            'gt': self._int64_feature(gtbox_label)
        })
        example = tf.train.Example(features=feature)
        self.writer.write(example.SerializeToString())
    def close(self):
        self.writer.close()

class Img2TFrecord(object):
    def __init__(self,args):
        self.anno_file = args.anno_file
        save_dir = args.save_dir
        #dataset_name = cfgs.DATASET_NAME #args.dataset_name
        assert args.dataset_name == cfgs.DATASET_NAME,'input data name should be equal to config name'
        dataset_name = args.dataset_name
        self.image_dir = args.image_dir
        save_name = args.save_name
        self.img_format = args.img_format
        save_path = os.path.join(save_dir,dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.property_file = os.path.join(save_path,'property_'+save_name+'.txt')
        save_name = save_name + '.tfrecord'
        record_save_path = os.path.join(save_path,save_name)
        self.record_w = DataToRecord(record_save_path)
        

    def rd_anotation(self,annotation):
        '''
        annotation: 1/img_01 1 ...
        '''
        img_dict = dict()
        annotation = annotation.strip().split()
        self.img_prefix = annotation[0]
        #gt
        self.label = string.atoi(annotation[1])
        #if int(self.label) != 0:
         #   return None
        #load image
        img_path = os.path.join(self.image_dir, self.img_prefix)
        if not os.path.exists(img_path):
            return None
        self.img_org = cv2.imread(img_path)
        if self.img_org is None:
            return None
        img_shape = self.img_org.shape[:2]
        #img = img[:,:,::-1]
        if cfgs.BIN_DATA:
            img_raw = open(img_path,'rb').read()
        self.img_name = img_path.split('/')[-1]
        img_dict['img_data'] = img_raw if cfgs.BIN_DATA else self.img_org
        img_dict['img_shape'] = img_shape
        img_dict['gt'] = self.label 
        img_dict['img_name'] = self.img_name
        return img_dict

    def transform_img(self):
        '''
        annotation: 1/img_01 0 ...
        '''
        auger_list=["Sequential", "Fliplr","AdditiveGaussianNoise","SigmoidContrast","Multiply"]
        trans = transform.Transform(img_auger_list=auger_list,class_num=cfgs.CLS_NUM)
        img_dict = dict()
        if self.img_org is None:
            print("aug img is None")
            return None
        img_aug = trans.aug_img(self.img_org)
        if not len(img_aug) >0:
            #print("aug box is None")
            return None
        img_data = img_aug[0]
        img_dict['img_data'] = img_data
        img_dict['img_shape'] = img_data.shape[:2]
        img_dict['gt'] = self.label
        img_dict['img_name'] = self.img_prefix[:-4]+'_aug'+self.img_format
        return img_dict

    def convert_widerface_to_tfrecord(self):
        failed_aug_path = open('aug_failed.txt','w')
        property_w = open(self.property_file,'w')
        anno_p = open(self.anno_file,'r')
        anno_lines = anno_p.readlines()
        total_img = 0
        dataset_img_num = len(anno_lines)
        cnt_failed = 0
        for count,tmp in enumerate(anno_lines):
            img_dict = self.rd_anotation(tmp)
            if img_dict is None:
                #print("the img path is none:",tmp.strip().split()[0])
                continue
            self.record_w.write_recore(img_dict)
            #label_show(img_dict,'bgr')
            total_img+=1
            if random.randint(0, 1) and not cfgs.BIN_DATA:
                img_dict = self.transform_img()
                if img_dict is None:
                    #print("the aug img path is none:",tmp.strip().split()[0])
                    failed_aug_path.write(tmp.strip().split()[0] +'\n')
                    cnt_failed+=1
                    continue
                self.record_w.write_recore(img_dict)
                #label_show(img_dict,'bgr')
                total_img+=1
            view_bar('Conversion progress', count + 1,dataset_img_num)
        print('\nConversion is complete!')
        print('total img:',total_img)
        print("aug failed:",cnt_failed)
        property_w.write("{},{}".format(cfgs.CLS_NUM,total_img))
        property_w.close()
        self.record_w.close()
        anno_p.close()
        failed_aug_path.close()

def label_show(img_dict,mode='rgb'):
    img = img_dict['img_data']
    if mode == 'rgb':
        img = img[:,:,::-1]
    img = np.array(img,dtype=np.uint8)
    gt = float(img_dict['gt'])
    #print("img",img.shape)
    #print("box",gt.shape)
    score_label = str("{:.2f}".format(gt))
    cv2.putText(img,score_label,(int(20),int(20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("img",img)
    cv2.waitKey(0)

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


if __name__ == '__main__':
    args = parms()
    dataset = args.dataset_name
    '''
    if 'Prison' in dataset:
        ct = Img2TFrecord(args)
        ct.convert_widerface_to_tfrecord()
    elif 'Mobile' in dataset:
        ct = Img2TFrecord(args)
        ct.convert_widerface_to_tfrecord()
    elif 'Face' in dataset:
        ct = Img2TFrecord(args)
        ct.convert_widerface_to_tfrecord()
    else:
        print("pleace select right dataset")
    '''
    ct = Img2TFrecord(args)
    ct.convert_widerface_to_tfrecord()
