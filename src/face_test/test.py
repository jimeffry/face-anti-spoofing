# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/12 14:09
#project: Face detect
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect testing caffe model
####################################################
import sys
from tqdm import tqdm
import os
os.environ['GLOG_minloglevel'] = '2'
import tools_matrix as tools
import cv2
import numpy as np
import argparse
import time
from Detector import FaceDetector_Opencv,MTCNNDet
from align import alignImg
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from imgpad import Img_Pad
sys.path.append(os.path.join(os.path.dirname(__file__),'../test'))
from get_model import Face_Anti_Spoof
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def args():
    parser = argparse.ArgumentParser(description="mtcnn caffe")
    parser.add_argument('--file-in',type=str,dest='file_in',default=None,\
                        help="the file input path")
    parser.add_argument('--min-size',type=int,dest='min_size',default=24,\
                        help="scale img size")
    parser.add_argument('--img-path',type=str,dest='img_path',default="test1.jpg",\
                        help="img1 saved path")
    parser.add_argument('--base-dir',type=str,dest='base_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--save-dir',type=str,dest='save_dir',default="./",\
                        help="images saved dir")
    parser.add_argument('--base-name',type=str,dest='base_name',default="videox",\
                        help="images saved dir")
    parser.add_argument('--cmd-type',type=str,dest='cmd_type',default="video",\
                        help="detect face from : video or txtfile")
    parser.add_argument('--save-dir2',type=str,dest='save_dir2',default=None,\
                        help="images saved dir")
    parser.add_argument('--crop-size',type=str,dest='crop_size',default='112,112',\
                        help="images saved size")
    parser.add_argument('--model-dir',type=str,dest='model_dir',default="../../models/",\
                        help="models saved dir")
    parser.add_argument('--gpu', default=0, type=int,help='which gpu to run')
    parser.add_argument('--load-epoch', default=0,dest='load_epoch', type=int,\
                        help='saved epoch num')
    return parser.parse_args()


def evalu_img(args):
    imgpath = args.img_path
    min_size = args.min_size
    model_dir = args.model_dir
    cv2.namedWindow("test")
    cv2.moveWindow("test",1400,10)
    threshold = np.array([0.3,0.3,0.7])
    base_name = "test_img"
    save_dir = './output'
    crop_size = [112,112]
    detect_model = MTCNNDet(min_size,threshold,model_dir)
    img = cv2.imread(imgpath)
    h,w = img.shape[:2]
    if cfgs.img_downsample and h > 1000:
        img = img_ratio(img,240)
    rectangles = detect_model.detectFace(img)
    #draw = img.copy()
    if len(rectangles)>0:
        points = np.array(rectangles)
        #print('rec shape',points.shape)
        points = points[:,5:]
        points_list = points.tolist()
        crop_imgs = alignImg(img,crop_size,points_list)
        for idx_cnt,img_out in enumerate(crop_imgs):
            savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
            #img_out = cv2.resize(img_out,(96,112))
            #cv2.imshow("test",img_out)
            cv2.imwrite(savepath,img_out)
        label_show(img,rectangles)
    else:
        print("No face detected")
    cv2.imshow("test",img)
    cv2.waitKey(0)
    #cv2.imwrite('test.jpg',draw)


def label_show(img,rectangles,pred_id):
    for rectangle in rectangles:
        score_label = "%s_face" %(cfgs.DATA_NAME[pred_id])
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        if len(rectangle) > 5:
            if cfgs.x_y:
                for i in range(5,15,2):
                    cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            else:
                rectangle = rectangle[5:]
                for i in range(5):
                    cv2.circle(img,(int(rectangle[i]),int(rectangle[i+5])),2,(0,255,0))
        
def sort_box(boxes_or):
    boxes = np.array(boxes_or)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = area.argsort()[::-1]
    #print(I)
    #print(boxes_or[0])
    idx = map(int,I)
    return boxes[idx[:]].tolist()

def img_ratio(img,img_h):
    h,w,c = img.shape
    ratio_ = float(h) / float(w)
    img_w = img_h / ratio_
    img_out = cv2.resize(img,(int(img_w),int(img_h)))
    return img_out

def mk_dirs(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def img_crop(img,bbox,imgw,imgh):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    if cfgs.box_widen:
        boxw = x2-x1
        boxh = y2-y1
        x1 = int(max(0,int(x1-3*boxw)))
        y1 = int(max(0,int(y1-4*boxh)))
        x2 = int(min(imgw,int(x2+3*boxw)))
        y2 = int(min(imgh,int(y2+4*boxh)))
    cropimg = img[y1:y2,x1:x2,:]
    return cropimg

def save_cropfromtxt(args):
    '''
    file_in: images path recorded
    base_dir: images locate in 
    save_dir: detect faces saved in
    fun: saved id images, save name is the same input image
    '''
    file_in = args.file_in
    #base_name = args.base_name
    #save_dir = args.save_dir
    #crop_size = args.crop_size
    #size_spl = crop_size.strip().split(',')
    #crop_size = [int(size_spl[0]),int(size_spl[1])]
    model_dir = args.model_dir
    min_size = args.min_size
    f_ = open(file_in,'r')
    failed_w = open('./output/failed_face3.txt','w')
    lines_ = f_.readlines()
    threshold = np.array([0.3,0.3,0.7])
    detect_model = MTCNNDet(min_size,threshold,model_dir) 
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    mk_dirs(save_dir)
    idx_cnt = 0 
    if cfgs.show:
        cv2.namedWindow("src")
        cv2.namedWindow("crop")
        cv2.moveWindow("crop",650,10)
        cv2.moveWindow("src",100,10)
    total_item = len(lines_)
    for i in tqdm(range(total_item)):
        line_1 = lines_[i]
        line_1 = line_1.strip()
        img_path = os.path.join(base_dir,line_1)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h,w = img.shape[:2]
        if cfgs.img_downsample and min(w,h) > 1000:
            img = img_ratio(img,240)
        line_s = line_1.split("/")  
        img_name = line_s[-1]
        new_dir = '/'.join(line_s[:-1]) 
        rectangles = detect_model.detectFace(img)
        if len(rectangles)> 0:
            idx_cnt+=1
            rectangles = sort_box(rectangles)
            '''
            if not cfgs.crop_org:
                points = np.array(rectangles)
                points = points[:,5:]
                points_list = points.tolist()
                points_list = [points_list[0]]
                img_out = alignImg(img,crop_size,points_list)
                img_out = img_out[0]
            else:
                img_out = img_crop(img,rectangles[0],img.shape[1],img.shape[0])
                #savepath = os.path.join(save_dir,str(idx_cnt)+".jpg")
                if cfgs.imgpad:
                    img_out = Img_Pad(img_out,crop_size)
                else:
                    img_out = cv2.resize(img_out,(crop_size[1],crop_size[0]))
            savepath = os.path.join(save_dir,line_1)
            '''
            cv2.imwrite(savepath,img_out)
            cv2.waitKey(10)
            #cv2.imwrite(savepath,img)
            if cfgs.show:
                label_show(img,rectangles)
                cv2.imshow("crop",img_out)
                cv2.waitKey(1000)
        else:
            failed_w.write(img_path)
            failed_w.write('\n')
            print("failed ",img_path)
        if cfgs.show:
            cv2.imshow("src",img)
            cv2.waitKey(10)
    failed_w.close()
    f_.close()


def save_cropfromvideo(args):
    '''
    file_in: input video file path
    base_name: saved images prefix name
    save_dir: saved images path
    fun: saved detect faces to dir
    '''
    file_in = args.file_in
    #base_name = args.base_name
    #save_dir = args.save_dir
    #save_dir2 = args.save_dir2
    #crop_size = args.crop_size
    #size_spl = crop_size.strip().split(',')
    #crop_size = [int(size_spl[0]),int(size_spl[1])]
    min_size = args.min_size
    model_dir = args.model_dir
    FaceAnti_model_dir = os.path.join(model_dir,cfgs.DATASET_NAME)
    model_path = os.path.join(FaceAnti_model_dir,cfgs.MODEL_PREFIX)
    FaceAnti_Model = Face_Anti_Spoof(model_path,args.load_epoch,cfgs.IMG_SIZE,args.gpu,layer='fc')
    caffe_mode_dir = os.path.join(model_dir,'FaceDetect')
    threshold = np.array([0.7,0.8,0.95])
    Detect_Model = MTCNNDet(min_size,threshold,caffe_mode_dir) 
    if file_in is None:
        v_cap = cv2.VideoCapture(0)
    else:
        v_cap = cv2.VideoCapture(file_in)
    #model_path = "../models/haarcascade_frontalface_default.xml"
    #detect_model = FaceDetector_Opencv(model_path)
    idx_cnt = 0 
    #mk_dirs(save_dir)
    #mk_dirs(save_dir2)
    if not v_cap.isOpened():
        print("field to open video")
    else:
        print("video frame num: ",v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_num = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_w = v_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_h = v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_cnt = 0
        class_dict = dict()
        pred_id_show = int(0)
        cv2.namedWindow("src")
        cv2.namedWindow("crop")
        cv2.moveWindow("crop",1400,10)
        cv2.moveWindow("src",10,10)
        while v_cap.isOpened():
            ret,frame = v_cap.read()
            if ret: 
                rectangles = Detect_Model.detectFace(frame)
            else:
                continue
            if len(rectangles)> 0:
                rectangles = sort_box(rectangles)
                frame_cnt+=1
                '''
                if cfgs.crop_org:
                    for bbox_one in rectangles:
                        idx_cnt+=1
                        img_out = img_crop(frame,bbox_one,frame_w,frame_h)
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #savepath = os.path.join(save_dir,line_1)
                        img_out = cv2.resize(img_out,(96,112))
                        cv2.imwrite(savepath,img_out)
                else:
                    points = np.array(rectangles)
                    points = points[:,5:]
                    points_list = points.tolist()
                    crop_imgs = alignImg(frame,crop_size,points_list)
                    for box_idx,img_out in enumerate(crop_imgs):
                        idx_cnt+=1
                        savepath = os.path.join(save_dir,base_name+'_'+str(idx_cnt)+".jpg")
                        #img_out = cv2.resize(img_out,(96,112))
                        #cv2.imshow("test",img_out)
                        cv2.imwrite(savepath,img_out)
                        cv2.waitKey(50)
                        if cfgs.box_widen:
                            savepath2 = os.path.join(save_dir2,base_name+'_'+str(idx_cnt)+".jpg")
                            img_widen = img_crop(frame,rectangles[box_idx],frame_w,frame_h)
                            cv2.imwrite(savepath2,img_widen)
                            cv2.waitKey(50)
                        print("crop num,",idx_cnt)
                '''
                img_verify = img_crop(frame,rectangles[0],frame_w,frame_h)
                tmp,pred_id = FaceAnti_Model.inference(img_verify)
                if frame_cnt == 10:
                    frame_cnt =0
                    for key_name in class_dict.keys():
                        if key_name in cfgs.DATA_NAME[1:]:
                            pred_id_show = int(1)
                            break
                        else:
                            pred_id_show = int(0)
                    class_dict.clear()
                else:
                    cur_cnt = class_dict.setdefault(cfgs.DATA_NAME[pred_id],0)
                    class_dict[cfgs.DATA_NAME[pred_id]] = cur_cnt+1
                label_show(frame,rectangles,pred_id_show)
                cv2.imshow('crop',img_verify)
            else:
                #print("failed ")
                pass
            cv2.imshow("src",frame)
            key_ = cv2.waitKey(10) & 0xFF
            if key_ == 27 or key_ == ord('q'):
                break
            #if fram_cnt == total_num:
             #   break
    v_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #main()
    parm = args()
    cmd_type = parm.cmd_type
    if cmd_type == 'txtfile':
        save_cropfromtxt(parm)
    elif cmd_type == 'video':
        save_cropfromvideo(parm)
    elif cmd_type == 'imgtest':
        evalu_img(parm)
    else:
        print("No cmd run")
