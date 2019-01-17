# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/24 15:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################
import numpy as np
import tensorflow as tf
import os
from image_preprocess import short_side_resize,img_resize
from image_preprocess import norm_data
from convert_data_to_tfrecord import label_show
import sys
from read_tfrecord import Read_Tfrecord
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfgs


def read_multi_rd(data_record_dir,fg_name,bg_name,batch_size,total_num,ratio=0.25):
    bg_batch = int(np.floor(batch_size * ratio))
    fg_batch = batch_size - bg_batch
    fg_rd = Read_Tfrecord(cfgs.DATASET_NAME,data_record_dir,fg_batch,total_num,fg_name)
    fg_name_batch, fg_img_batch, fg_label_batch = fg_rd.next_batch()
    bg_rd = Read_Tfrecord(cfgs.DATASET_NAME,data_record_dir,bg_batch,total_num,bg_name)
    bg_name_batch, bg_img_batch, bg_label_batch = bg_rd.next_batch()
    images_batch = tf.concat([fg_img_batch,bg_img_batch], 0, name="concat/image")
    labels_batch = tf.concat([fg_label_batch,bg_label_batch],0,name="concat/label")
    return images_batch,labels_batch

if __name__ == '__main__':
    img_dict = dict()
    sess = tf.Session()
    img_batch, gtboxes_and_label_batch = read_multi_rd('../../data','fg','bg',4,5000)
    '''
    tfrd = Read_Tfrecord('Prison','../../data',4,True)
    img_name_batch, img_batch, gtboxes_and_label_batch = tfrd.next_batch()
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for i in range(10):
            print("idx",i)
            img,gt = sess.run([img_batch,gtboxes_and_label_batch])
            print("img",np.shape(img))
            print('gt',np.shape(gt),gt[0])
            #print('name:',name)
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