# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import sys 
import os 
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
import config as cfgs

def parms():
    parser = argparse.ArgumentParser(description='gen img lists for confusion matrix test')
    parser.add_argument('--img-dir',type=str,dest="img_dir",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--dir2-in',type=str,dest="dir2_in",default='./',\
                        help='the directory should include 2 more Picturess')
    parser.add_argument('--file-in',type=str,dest="file_in",default="train.txt",\
                        help='img paths saved file')
    parser.add_argument('--save-dir',type=str,dest="save_dir",default='./',\
                        help='img saved dir')
    parser.add_argument('--out-file',type=str,dest="out_file",default="train.txt",\
                        help='out img paths saved file')
    parser.add_argument('--cmd-type',type=str,dest="cmd_type",default="gen_label",\
                        help='which code to run: gen_label_pkl, gen_label,merge,gen_filepath_1dir,save_idimgfrom_txt,\
                        hist, imgenhance,gen_idimgfrom_dir,gen_filepath_2dir')
    parser.add_argument('--file2-in',type=str,dest="file2_in",default="train2.txt",\
                        help='label files')
    return parser.parse_args()

def max_length_limitation(length, length_limitation):
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)

def short_side_resize(img_tensor, target_shortside_len, length_limitation=1200):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, length_limitation=1200):
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor


def norm_data(img):
    img[:,:,0] = img[:,:,0]- cfgs.PIXEL_MEAN[0] # R
    img[:,:,1] = img[:,:,1]- cfgs.PIXEL_MEAN[1] # G
    img[:,:,2] = img[:,:,2]- cfgs.PIXEL_MEAN[2] # B
    img = img/cfgs.PIXEL_NORM
    return img.astype(np.float32)

def de_norm_data(img):
    img = img * cfgs.PIXEL_NORM
    img[:,:,0] = img[:,:,0]+ cfgs.PIXEL_MEAN[0]
    img[:,:,1] = img[:,:,1]+ cfgs.PIXEL_MEAN[1] # G
    img[:,:,2] = img[:,:,2]+ cfgs.PIXEL_MEAN[2] # B
    return img.astype(np.uint8)

def generate_list_from_dir(dirpath,out_file):
    '''
    dirpath: saved images path
            "dirpath/id_num/image1.jpg"
    return: images paths txtfile
            "id_num/img1.jpg"
    '''
    f_w = open(out_file,'w')
    files = os.listdir(dirpath)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    file_name = []
    total_cnt = 0
    label = 0
    for file_cnt in files:
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        for img_one in imgs:
            img_path = os.path.join(file_cnt,img_one)
            total_cnt+=1
            f_w.write("{} {}\n".format(img_path,label))
    print("total img ",total_cnt)
    cnt = 0
    f_w.close()

def gen_filefromdir(base_dir,txt_file):
    '''
    base_dir: saved images path
        "base_dir/image.jpg"
    return: "image1.jpg"
    '''
    f_w = open(txt_file,'w')
    files = os.listdir(base_dir)
    total_ = len(files)
    label = 1
    print("total id ",len(files))
    for file_cnt in files:
        f_w.write("{} {}\n".format(file_cnt,label))
    f_w.close()

def merge2trainfile(file1,file2,file_out):
    '''
    file1: saved image  paths
    file2: saved image paths
    return: "file1 file2" 
    '''
    f1 = open(file1,'r')
    f2 = open(file2,'r')
    f_out = open(file_out,'w')
    id_files = f1.readlines()
    imgs = f2.readlines()
    max_num = int(max(len(id_files),len(imgs)))
    for i in range(max_num):
        if i <= len(id_files)-1:
            f_out.write(id_files[i].strip())
            f_out.write("\n")
        if i <= len(imgs)-1:
            f_out.write(imgs[i].strip())
            f_out.write("\n")
    f1.close()
    f2.close()
    f_out.close()
    print("over")

if __name__ == "__main__":
    args = parms()
    txt_file = args.file_in
    #get_from_txt(txt_file)
    img_dir = args.img_dir
    save_dir = args.save_dir
    file2_in = args.file2_in
    out_file = args.out_file
    cmd_type = args.cmd_type
    
    if cmd_type == 'merge':
        merge2trainfile(txt_file,file2_in,out_file)
    elif cmd_type == 'gen_filepath_1dir':
        gen_filefromdir(img_dir,out_file)
    elif cmd_type == 'gen_filepath_2dir':
        generate_list_from_dir(img_dir,out_file)
