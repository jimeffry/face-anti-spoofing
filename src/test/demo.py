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
import tensorflow as tf 
import numpy as np 
import time 
import cv2
import argparse
import os 
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
from face_anti import Face_Anti_Spoof


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--out-file',type=str,dest='out_file',default='None',\
                        help="the file output path")
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./base_dir",\
                        help="images saved dir")
    parser.add_argument('--caffemodel',type=str,dest='m_path',default="../models/deploy.caffemodel",\
                        help="caffe model path")
    parser.add_argument('--tf-model',type=str,dest='tf_model',default="../../models/",\
                        help="models saved dir")
    parser.add_argument('--gpu', default='0', type=str,help='which gpu to run')
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./saved_dir",\
                        help="images saved dir")
    parser.add_argument('--failed-dir',type=str,dest='failed_dir',default="./failed_dir",\
                        help="fpr saved dir")
    parser.add_argument('--load-epoch', default='0',dest='load_epoch', type=str,help='saved epoch num')
    parser.add_argument('--cmd-type', default="dbtest",dest='cmd_type', type=str,\
                        help="which code to run: videotest,imgtest ")
    return parser.parse_args()

def test_img(args):
    '''
    '''
    img_path1 = args.img_path1
    model_dir = args.tf_model
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX) + '-'+args.load_epoch
    Model = Face_Anti_Spoof(model_path,cfgs.IMG_SIZE,args.gpu)
    img_data1 = cv2.imread(img_path1)
    if img_data1 is None:
        print('img is none')
        return None
    fram_h,fram_w = img_data1.shape[:2]
    tmp,pred_id = Model.inference(img_data1)
    print("pred",tmp)
    score_label = "{}".format(cfgs.DATA_NAME[pred_id])
    cv2.putText(img_data1,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img_data1)
    cv2.waitKey(0)

def display(img,id):
    fram_h,fram_w = img.shape[:2]
    score_label = "{}".format(cfgs.DATA_NAME[id])
    cv2.putText(img,score_label,(int(20),int(30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img)
    cv2.waitKey(0)

def evalue(args):
    '''
    calculate the tpr and fpr for all classes
    R = tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    P = tp/(tp+fp)
    '''
    file_in = args.file_in
    result_out = args.out_file
    img_dir = args.base_dir
    model_dir = args.tf_model
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX) + '-'+args.load_epoch
    Model = Face_Anti_Spoof(model_path,cfgs.IMG_SIZE,args.gpu)
    if file_in is None:
        print("input file is None",file_in)
        return None
    file_rd = open(file_in,'r')
    file_wr = open(result_out,'w')
    file_cnts = file_rd.readlines()
    total_num = len(file_cnts)
    statistics_dic = dict()
    for name in cfgs.DATA_NAME:
        statistics_dic[name+'_tpr'] = 0
        statistics_dic[name+'_fpr'] = 0
        statistics_dic[name] = 0
    for i in tqdm(range(total_num)):
        item_cnt = file_cnts[i]
        item_spl = item_cnt.strip().split()
        img_path,real_label = item_spl[:]
        img_path = os.path.join(img_dir,img_path)
        img_data = cv2.imread(img_path)
        if img_data is None:
            print('img is none',img_path)
            continue
        probility,pred_id = Model.inference(img_data)
        pred_name = cfgs.DATA_NAME[pred_id]
        real_name = cfgs.DATA_NAME[int(real_label)]
        statistics_dic[real_name] +=1
        if pred_id == int(real_label):
            statistics_dic[pred_name+'_tpr'] +=1
        else:
            statistics_dic[pred_name+'_fpr'] +=1
        if cfgs.ShowImg:
            display(img_data,pred_id)
    for key_name in cfgs.DATA_NAME:
        tp_fn = statistics_dic[key_name]
        tp = statistics_dic[key_name+'_tpr']
        fp = statistics_dic[key_name+'_fpr']
        fp_tn = total_num - tp_fn
        tpr = float(tp) / tp_fn
        fpr = float(fp) / fp_tn
        precision = float(tp) / (tp+fp+1)
        statistics_dic[key_name+'_tpr'] = tpr
        statistics_dic[key_name+'_fpr'] = fpr
        statistics_dic[key_name+'_P'] = precision
        file_wr.write('{} result is: tp_fn-{},fp_tn-{},tp-{},fp-{}\n'.format(key_name,\
                        tp_fn,fp_tn,tp,fp))
        file_wr.write('\t tpr:{}  fpr:{}  Precision:{}\n'.format(tpr,fpr,precision))
    file_rd.close()
    file_wr.close()




def test_video(args):
    model_dir = args.tf_model
    file_in = args.file_in
    model_dir = args.tf_model
    model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(model_dir,cfgs.MODEL_PREFIX) + '-'+args.load_epoch
    Model = Face_Anti_Spoof(model_path,cfgs.IMG_SIZE,args.gpu)
    if file_in is None:
        v_cap = cv2.VideoCapture(11)
    else:
        v_cap = cv2.VideoCapture(file_in)
    cv2.namedWindow("video")
    cv2.moveWindow("video",650,10)
    if not v_cap.isOpened():
        print("field to open video")
    else:
        if file_in is not None:
            print("video frame num: ",v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_num = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_num > 100000:
                total_num = 30000
        else:
            total_num = 100000
        #record_w.write("the video has total num: %d \n" % total_num)
        #if frame_num is not None:
         #   v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        fram_w = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fram_h = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fram_cnt = 0
        while v_cap.isOpened():
            ret,frame = v_cap.read()
            fram_cnt+=1
            sys.stdout.write("\r>> deal with %d / %d" % (fram_cnt,total_num))
            sys.stdout.flush()
            if ret: 
                t = time.time()
                _,pred_id = Model.inference(frame)
                t_det = time.time() - t
            else:
                continue
            score_label = str("{:.2f}".format(pred_id))
            #score_label = str(1.0)
            cv2.putText(frame,score_label,(int(fram_w-20),int(20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.imshow("video",frame)
            key_ = cv2.waitKey(10) & 0xFF
            if key_ == 27 or key_ == ord('q'):
                break
            if fram_cnt == total_num:
                break

if __name__ == '__main__':
    parms = args()
    cmd_type = parms.cmd_type
    if cmd_type in 'imgtest':
        test_img(parms)
    elif cmd_type in 'videotest':
        test_video(parms)
    elif cmd_type in 'filetest':
        evalue(parms)
    else:
        print('Please input right cmd')