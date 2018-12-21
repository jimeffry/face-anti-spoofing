# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/10 15:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from image_preprocess import short_side_resize
from image_preprocess import norm_data
from convert_data_to_tfrecord import label_show
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfgs

class Read_Tfrecord(object):
    def __init__(self,dataset_name,data_dir,batch_size, is_training):
        if dataset_name not in ['Prison', 'WiderFace']:
            raise ValueError('dataSet name must be in pascal, coco Prison')
        if is_training:
            tfrecord_file = os.path.join(data_dir, dataset_name,'train.tfrecord')
        else:
            tfrecord_file = os.path.join(data_dir, dataset_name ,'test.tfrecord')
        print('tfrecord path is -->', os.path.abspath(tfrecord_file))
        self.batch_size = batch_size
        #filename_tensorlist = tf.train.match_filenames_once(pattern)
        #self.filename_queue = tf.train.string_input_producer(filename_tensorlist)
        self.filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
        self.reader = tf.TFRecordReader()

    def read_single_example_and_decode(self):
        _, serialized_example = self.reader.read(self.filename_queue)
        features = tf.parse_single_example(
            serialized=serialized_example,
            features={
                'img_name': tf.FixedLenFeature([], tf.string),
                'img_height': tf.FixedLenFeature([], tf.int64),
                'img_width': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }
        )
        img_name = features['img_name']
        img_height = tf.cast(features['img_height'], tf.int32)
        img_width = tf.cast(features['img_width'], tf.int32)
        #print("begin to decode")
        img = tf.image.decode_jpeg(features['img'],channels=3)
        img = tf.reshape(img, shape=[img_height, img_width, 3])
        gt_labels = tf.decode_raw(features['gt'], tf.int32)
        gt_labels = tf.reshape(gt_labels, [-1])
        return img_name, img, gt_labels

    def process_img(self,img):
        if cfgs.IMG_LIMITATE:
            img = short_side_resize(img_tensor=img,
                                        target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                        length_limitation=cfgs.IMG_MAX_LENGTH)
        img = tf.py_func(norm_data,[img],tf.float32)
        img.set_shape([None,None,3])
        return img

    def next_batch(self):
        img_name, img_raw, gt_label = self.read_single_example_and_decode()
        img_data = self.process_img(img_raw)
        #print("begin to batch")
        img_name_batch, img_batch, gt_label_batch = \
            tf.train.batch(
                       [img_name, img_data, gt_label],
                       batch_size=self.batch_size,
                       capacity=1,
                       num_threads=1,
                       dynamic_pad=True)
        return img_name_batch, img_batch, gt_label_batch


if __name__ == '__main__':
    img_dict = dict()
    sess = tf.Session()
    tfrd = Read_Tfrecord('Prison','../../data',1,True)
    img_name_batch, img_batch, gtboxes_and_label_batch = tfrd.next_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for i in range(10):
            print("idx",i)
            img,gt,name,obg = sess.run([img_batch,gtboxes_and_label_batch,img_name_batch])
            print("img",np.shape(img))
            print('gt',np.shape(gt))
            print('name:',name)
            #print('num_obg:',obg)
            print('data',img[0,5,:5,0])
            img_dict['img_data'] = img[0]
            img_dict['gt'] = gt[0]
            #label_show(img_dict)
    except tf.errors.OutOfRangeError:
        print("Over！！！")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()