# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/01/17 14:09
#project: Face recognize
#company: Senscape
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  
####################################################
import numpy as np 
import cv2

def get_by_ratio(x,new_x,y):
    ratio = x / float(new_x)
    new_y = y / ratio
    return np.floor(new_y)

def Img_Pad(img,crop_size):
    '''
    img: input img data
    crop_size: [h,w]
    '''
    img_h,img_w = img.shape[:2]
    d_h,d_w = crop_size
    pad_l,pad_r,pad_u,pad_d = [0,0,0,0]
    if img_w > d_w or img_h > d_h :
        if img_h> img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w < d_w or img_h < d_h:
        if img_h < img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            #print("debug1",new_h,new_w)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                #print("debug2",new_h,new_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            #print("up",new_h,new_w)
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w==d_w and img_h==d_h:
        img_out = img
    if not [pad_l,pad_r,pad_u,pad_d] == [0,0,0,0] :
        color = [255,255,255]
        #print("padding",[pad_l,pad_r,pad_u,pad_d])
        img_out = cv2.copyMakeBorder(img_out,top=int(pad_u),bottom=int(pad_d),left=int(pad_l),right=int(pad_r),\
                                    borderType=cv2.BORDER_CONSTANT,value=color) #BORDER_REPLICATE
    return img_out