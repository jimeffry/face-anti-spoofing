# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/06/28 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description   opencv face detector
####################################################
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
sys.path.append('.')
sys.path.append('/home/lxy/caffe/python')
os.environ['GLOG_minloglevel'] = '2'
import tools_matrix as tools
import caffe
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs as config


class FaceDetector_Opencv(object):
    def __init__(self,model_path):
        self.detection_model = cv2.CascadeClassifier(model_path)
        print("load model over")

    def detectFace(self,img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detection_model.detectMultiScale(gray_image, 1.3, 5,minSize=(130,130),maxSize=(900,900))
        #results = detector.detect_face(img)
        boxes = []
        for face_coordinates in faces:
            boxes.append(face_coordinates)
        if len(boxes)>0:
            boxes = np.asarray(boxes)
            boxes[:,2] = boxes[:,0] +boxes[:,2]
            boxes[:,3] = boxes[:,1] +boxes[:,3]
            return boxes
        else:
            return []

    def draw_box(self,img,box,color=(255,0,0)):
        #(row,col,cl) = np.shape(img)
        #b = board_img(box,col,row)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), color)

    def add_label(self,img,bbox,label,color=(255,0,0)):
        num = bbox.shape[0]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale =1
        thickness = 1
        for i in range(num):
            x1,y1,w,h = int(bbox[i,0]),int(bbox[i,1]),int(bbox[i,2]),int(bbox[i,3])
            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
            #score_label = str('{:.2f}'.format(bbox[i,4]))
            score_label = label
            size = cv2.getTextSize(score_label, font, font_scale, thickness)[0]
            if y1-int(size[1]) <= 0:
                #cv2.rectangle(img, (x1, y2), (x1 + int(size[0]), y2+int(size[1])), color)
                cv2.putText(img, score_label, (x1,y2+size[1]), font, font_scale, color, thickness)
            else:
                #cv2.rectangle(img, (x1, y1-int(size[1])), (x1 + int(size[0]), y1), (255, 0, 0))
                cv2.putText(img, score_label, (x1,y1), font, font_scale, color, thickness)

class MTCNNDet(object):
    def __init__(self,min_size,threshold,model_dir):
        self.test = 1
        self.load = 1
        self.load_model(model_dir)
        self.min_size = min_size
        self.threshold = threshold
        caffe.set_device(0)
        caffe.set_mode_gpu()
    def load_model(self,model_dir):
        test_ = self.test
        load_ = self.load
        if load_ :
            deploy = '12net.prototxt'
            caffemodel = '12net.caffemodel'
        else:
            deploy = 'PNet.prototxt'
            caffemodel = 'PNet.caffemodel'
        deploy = os.path.join(model_dir,deploy)
        caffemodel = os.path.join(model_dir,caffemodel)
        self.net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
        if load_:
            deploy = '24net.prototxt'
            caffemodel = '24net.caffemodel'
        else:
            deploy = 'RNet.prototxt'
            caffemodel = 'RNet.caffemodel'
        deploy = os.path.join(model_dir,deploy)
        caffemodel = os.path.join(model_dir,caffemodel)
        self.net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)
        if load_:
            deploy = '48net.prototxt'
            caffemodel = '48net.caffemodel'
        else:
            deploy = "onet.prototxt"
            caffemodel = "onet.caffemodel"
        deploy = os.path.join(model_dir,deploy)
        caffemodel = os.path.join(model_dir,caffemodel)
        self.net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

    def PNet_(self,caffe_img):
        origin_h,origin_w,ch = caffe_img.shape
        scales = tools.calculateScales(caffe_img,self.min_size)
        out = []
        for scale in scales:
            hs = int(origin_h*scale)
            ws = int(origin_w*scale)
            #print(hs,ws)
            if self.test:
                scale_img = cv2.resize(caffe_img,(ws,hs))
                scale_img = np.swapaxes(scale_img, 0, 2)
                self.net_12.blobs['data'].reshape(1,3,ws,hs)
            else:
                scale_img = cv2.resize(caffe_img,(ws,hs))
                scale_img = np.transpose(scale_img, (2,0,1))
                self.net_12.blobs['data'].reshape(1,3,hs,ws)
            scale_img = np.asarray(scale_img,dtype=np.float32)
            self.net_12.blobs['data'].data[...]=scale_img
            out_ = self.net_12.forward()
            out.append(out_)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i]['prob1'][0][1]
            if self.test:
                roi      = out[i]['conv4-2'][0]
                #roi      = out[i]['conv4_2'][0]
            else:
                roi      = out[i]['conv4_2'][0]
            out_h,out_w = cls_prob.shape
            out_side = max(out_h,out_w)
            rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,self.threshold[0])
            rectangles.extend(rectangle)
        rectangles_box = tools.NMS(rectangles,0.7,'iou')
        return rectangles_box

    def RNet_(self,caffe_img,rectangles):
        origin_h,origin_w,ch = caffe_img.shape
        self.net_24.blobs['data'].reshape(len(rectangles),3,24,24)
        crop_number = 0
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(24,24))
            if self.test:
                scale_img = np.swapaxes(scale_img, 0, 2)
            else:
                scale_img = np.transpose(scale_img,(2,0,1))
            scale_img = np.asarray(scale_img,dtype=np.float32)
            self.net_24.blobs['data'].data[crop_number] =scale_img
            crop_number += 1
        out = self.net_24.forward()
        cls_prob = out['prob1']
        if self.test:
            roi_prob = out['conv5-2']
            #roi_prob = out['bbox_fc']
            #pts_prob = out['landmark_fc']
        else:
            roi_prob = out['bbox_fc']
        rectangles_box = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,self.threshold[1])
        #rectangles_box = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,self.threshold[1])
        return rectangles_box

    def ONet_(self,caffe_img,rectangles):
        origin_h,origin_w,ch = caffe_img.shape
        self.net_48.blobs['data'].reshape(len(rectangles),3,48,48)
        crop_number = 0
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(48,48))
            if self.test:
                scale_img = np.swapaxes(scale_img, 0, 2)
            else:
                scale_img = np.transpose(scale_img,(2,0,1))
            scale_img = np.asarray(scale_img,dtype=np.float32)
            self.net_48.blobs['data'].data[crop_number] =scale_img
            crop_number += 1
        out = self.net_48.forward()
        cls_prob = out['prob1']
        if self.test:
            roi_prob = out['conv6-2']
            pts_prob = out['conv6-3']
            #roi_prob = out['bbox_fc']
            #pts_prob = out['landmark_fc']
        else:
            roi_prob = out['bbox_fc']
            pts_prob = out['landmark_fc']
        rectangles_box = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,self.threshold[2])
        return rectangles_box

    def detectFace(self,img):
        #img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        caffe_img = (img.copy()-127.5)/128
        #origin_h,origin_w,ch = caffe_img.shape
        t = time.time()
        rectangles = self.PNet_(caffe_img)
        if config.onet_out:
            rectangles_back = rectangles
        if len(rectangles)==0 or config.pnet_out:
            return rectangles
        if config.time:
            print("Pnet proposals ",len(rectangles))
        t1 = time.time()-t
        t = time.time() 
        rectangles = self.RNet_(caffe_img,rectangles)
        t2 = time.time()-t
        t = time.time()
        if len(rectangles)==0 or config.rnet_out:
            return rectangles
        if config.onet_out:
            rectangles = self.ONet_(caffe_img,rectangles_back)
        else:
            rectangles = self.ONet_(caffe_img,rectangles)
        t3 = time.time()-t
        if config.time:
            print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet{:.3f}'.format(t1, t2,t3))
        return rectangles
