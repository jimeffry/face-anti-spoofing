# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import sys 
import os 
import csv
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfgs

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
    parser.add_argument('--base-label',type=int,dest="base_label",default=0,\
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

def img_resize(img_tensor, img_size=[224,224]):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor,img_size)
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
    '''
    img: should be rgb mode
    '''
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

def generate_list_from_dir(dirpath,out_file,label_num):
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
    label = label_num
    for file_cnt in files:
        '''
        if 'a_print_fg' in file_cnt or 'a_hand_fg' in file_cnt :
            label = 2
        elif 'a_mobile_fg' in file_cnt:
            label = 1
        else:
            label = 0
        '''
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        for img_one in imgs:
            #if len(img_one.strip()) < 9:
             #   continue
            img_path = os.path.join(file_cnt,img_one)
            total_cnt+=1
            f_w.write("{} {}\n".format(img_path,label))
    print("total img ",total_cnt)
    cnt = 0
    f_w.close()

def gen_filefromdir(base_dir,txt_file,label_num):
    '''
    base_dir: saved images path
        "base_dir/image.jpg"
    return: "image1.jpg"
    '''
    f_w = open(txt_file,'w')
    files = os.listdir(base_dir)
    total_ = len(files)
    label = label_num
    print("total id ",len(files))
    img_dir = base_dir.strip().split('/')[-1]
    for file_cnt in files:
        f_w.write("{} {}\n".format(os.path.join(img_dir,file_cnt),label))
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
    keys = id_files[0].strip().split()
    print(len(keys))
    keys = ','.join(keys)
    f_out.write("{}\n".format(keys))
    for i in range(1,max_num):
        file1 = id_files[i].strip()
        file2 = imgs[i-1].strip()
        f_s1 = file1.split()
        f_s2 = file2.split()
        if f_s1[0]==f_s2[0]:
            sub_name = '/'.join([f_s2[1],f_s2[0]])
            sub_txt = ','.join(f_s1[1:])
            sub_txt = ','.join([sub_name,sub_txt])
            f_out.write("{}\n".format(sub_txt))
        else:
            print("not equal",f_s1[0],f_s2[0])
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def merge2change(file1,file2,file_out,label_num):
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
            path = imgs[i].strip().split()[0]
            f_out.write("{} {}\n".format(path,label_num))
    f1.close()
    f2.close()
    f_out.close()
    print("over")

def generate_train_label(file_in,fileout):
    '''
    file_in: input_label csv file
    fileout: ouput train file
    '''
    f_in = open(file_in,'rb')
    f_out = open(fileout,'w')
    record = open('./output/record.txt','w')
    reader = csv.DictReader(f_in)
    label_keys = cfgs.FaceProperty
    #beard_keys = ['No_Beard','Mustache','Goatee','5_o_Clock_Shadow']
    #hair_keys = ['Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair']
    #head_keys = ['Bangs','Bald']
    cnt_dict = dict()
    cnt_err=0
    for f_item in reader:
        #print(f_item.keys())
        tmp_label = []
        img_name = f_item['filename']
        tmp_label.append(img_name)
        '''
        #beard
        label_beard = '4'
        for tmp_key in beard_keys:
            if int(f_item[tmp_key])==1:
                label_beard = str(beard_keys.index(tmp_key))
                cur_cnt = cnt_dict.setdefault(tmp_key,0)
                cnt_dict[tmp_key] = cur_cnt+1
                break
        if label_beard == '4':
            cur_cnt = cnt_dict.setdefault('has_beard',0)
            cnt_dict['has_beard'] = cur_cnt+1
        tmp_label.append(label_beard)
        #hair
        label_hair = '0'
        for tmp_key in hair_keys:
            if int(f_item[tmp_key])==1:
                label_hair = str(hair_keys.index(tmp_key)+1)
                cur_cnt = cnt_dict.setdefault(tmp_key,0)
                cnt_dict[tmp_key] = cur_cnt+1
                break
        if label_hair=='0':
            #cnt_dict['other_hair']+=1
            cur_cnt = cnt_dict.setdefault('other_hair',0)
            cnt_dict['other_hair'] = cur_cnt+1
        tmp_label.append(label_hair)
        #head
        label_head = '0'
        for tmp_key in head_keys:
            if int(f_item[tmp_key])==1:
                label_head = str(head_keys.index(tmp_key)+1)
                cur_cnt = cnt_dict.setdefault(tmp_key,0)
                cnt_dict[tmp_key] = cur_cnt+1
                break
        if label_head == '0':
            cur_cnt = cnt_dict.setdefault('normal_head',0)
            cnt_dict['normal_head'] = cur_cnt+1
        tmp_label.append(label_head)
        '''
        #other property
        for tmp_key in label_keys:
            if int(f_item[tmp_key])==1:
                tmp_label.append('1')
                cur_cnt = cnt_dict.setdefault(tmp_key+'_p',0)
                cnt_dict[tmp_key+'_p'] = cur_cnt +1
            else:
                tmp_label.append('0')
                cur_cnt = cnt_dict.setdefault(tmp_key+'_n',0)
                cnt_dict[tmp_key+'_n'] = cur_cnt +1
        #if int(f_item['Arched_Eyebrows'])==1 and int(f_item['Bushy_Eyebrows'])==1:
         #   cnt_err+=1
        tmp_label = ','.join(tmp_label)
        f_out.write("{}\n".format(tmp_label))
        #print(f_item['filename'])
    f_in.close()
    f_out.close()  
    #print(cnt_err)
    for key in cnt_dict.keys():
        record.write(key+' :'+str(cnt_dict[key])+'\n')

if __name__ == "__main__":
    args = parms()
    txt_file = args.file_in
    #get_from_txt(txt_file)
    img_dir = args.img_dir
    save_dir = args.save_dir
    file2_in = args.file2_in
    out_file = args.out_file
    cmd_type = args.cmd_type
    label = args.base_label
    if cmd_type == 'merge':
        merge2trainfile(txt_file,file2_in,out_file)
    elif cmd_type == 'gen_filepath_1dir':
        gen_filefromdir(img_dir,out_file,label)
    elif cmd_type == 'gen_filepath_2dir':
        generate_list_from_dir(img_dir,out_file,label)
    elif cmd_type in 'merge2change':
        merge2change(txt_file,file2_in,out_file,label)
    elif cmd_type == 'gen_trainlabel':
        generate_train_label(txt_file,out_file)
