# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/5 17:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################
from easydict import EasyDict as edict

def load_property(property_file):
  prop = edict()
  for line in open(property_file,'r'):
    vec = line.strip().split(',')
    assert len(vec)==2
    prop.cls_num = int(vec[0])
    prop.img_nums = int(vec[1])
  return prop