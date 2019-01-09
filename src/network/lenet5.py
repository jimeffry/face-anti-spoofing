# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/1/8 10:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################
import tensorflow as tf 
import tensorflow.contrib.layers as tfc # group_norm,conv2d,l2_regularizer,max_pool2d,avg_pool2d
import sys
import os 
#sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
#import config as cfgs

def Conv_block(data_in,kernel_size,filter_num,**kargs):
    conv_stride = kargs.get('conv_stride',1)
    #filter_num = kargs.get('filter_num',32)
    name_scope = kargs.get('name','res_base')
    w_regular = kargs.get('w_regular',None)
    train_fg = kargs.get('train_fg',True)
    bn_use = kargs.get('bn_use',True)
    group_num = kargs.get('group_num',32)
    eps = kargs.get('eps',1e-05)
    relu_type = kargs.get('relu_type',None)
    with tf.variable_scope(name_scope):
        bn_out = tfc.conv2d(data_in,filter_num,kernel_size,conv_stride,activation_fn=None,\
                            trainable=train_fg,weights_regularizer=w_regular,scope='%s_conv' % name_scope)
        #bn_out = tfc.group_norm(conv_out,group_num,epsilon=eps,scope='%s_bn' % name_scope)
        if bn_use:
            #bn_out = tfc.layer_norm(bn_out,scope='%s_bn' % name_scope)
            bn_out = tfc.batch_norm(bn_out,scope='%s_bn' % name_scope)
        if relu_type == 'relu':
            act_out = tf.nn.relu(bn_out,name='%s_relu' % name_scope)
        elif relu_type == 'relu6':
            act_out = tf.nn.relu6(bn_out,name='%s_relu6' % name_scope)
        elif relu_type == 'leaky_relu':
            act_out = tf.nn.leaky_relu(bn_out,name='%s_prelu' % name_scope)
        elif relu_type == None:
            act_out =None
        if act_out is None:
            return bn_out
        else:
            return act_out
def Max_pool2d(data_in,ker_size,step,name):
    return tfc.max_pool2d(data_in,ker_size,step,'SAME',scope=name)

def Lenet5(input_data,**kargs):
    conv1 = Conv_block(input_data,3,32,name='conv1',**kargs)
    poo1 = Max_pool2d(conv1,2,2,'pool1')
    conv2 = Conv_block(poo1,3,64,name='conv2',**kargs)
    pool2 = Max_pool2d(conv2,2,2,'pool2')
    conv3 = Conv_block(pool2,3,128,name='conv3',**kargs)
    pool3 = Max_pool2d(conv3,2,2,'pool3')
    conv4 = Conv_block(pool3,3,256,name='conv4',**kargs)
    pool4 = Max_pool2d(conv4,2,2,'pool4')
    conv5 = Conv_block(pool4,3,512,name='conv5',**kargs)
    pool5 = Max_pool2d(conv5,2,2,'pool5')
    return pool5

def get_symble(input_image,**kargs):
    w_decay = kargs.get('w_decay',1e-5)
    net_name = kargs.get('net_name','lenet5')
    class_num = kargs.get('class_num',93)
    train_fg = kargs.get('train_fg',True)
    w_r = tfc.l2_regularizer(w_decay)
    with tf.variable_scope(net_name):
        pool5 = Lenet5(input_image,w_regular=w_r,relu_type='relu',**kargs)
        flat = tfc.flatten(pool5,scope='flat')
        fc1 = tfc.fully_connected(flat,1024,activation_fn=tf.nn.relu,trainable=train_fg,\
                                weights_regularizer=w_r,scope='fc1')
        dp = tfc.dropout(fc1,keep_prob=0.5,is_training=train_fg,scope='drop_out')
        fc2 = tfc.fully_connected(dp,class_num,activation_fn=None,trainable=train_fg,\
                                weights_regularizer=w_r,scope='fc2')
        return fc2

if __name__=='__main__':
    graph = tf.Graph()
    with graph.as_default():
        img = tf.ones([64,112,112,3])
        model = get_symble(img,class_num=93,net_name='lenet5',train_fg=True)
        sess = tf.Session()
        summary = tf.summary.FileWriter('/home/lxy/Develop/Center_Loss/git_prj/face-anti-spoofing/logs/',sess.graph)