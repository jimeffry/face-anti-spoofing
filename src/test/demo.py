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
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
from face_anti import Face_Anti_Spoof


def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default='None',\
                        help="the file input path")
    parser.add_argument('--data-file',type=str,dest='data_file',default='None',\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=50,\
                        help="scale img size")
    parser.add_argument('--img-path1',type=str,dest='img_path1',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--img-path2',type=str,dest='img_path2',default="test2.jpg",\
                        help="scale img size")
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
    score_label = str("{:.2f}".format(pred_id))
    cv2.putText(img_data1,score_label,(int(fram_w-20),int(20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow("video",img_data1)
    cv2.waitKey(0)

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
    else:
        print('Please input right cmd')