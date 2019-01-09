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
import tensorflow as tf 
import tensorflow.contrib.layers as tfc # group_norm,conv2d,l2_regularizer,max_pool2d,avg_pool2d
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
import config

def Conv_block(data_in,kernel_size,**kargs):
        conv_stride = kargs.get('conv_stride',1)
        filter_num = kargs.get('filter_num',32)
        name_scope = kargs.get('name','res_base')
        w_regular = kargs.get('w_regular',None)
        train_fg = kargs.get('train_fg',True)
        bn_use = kargs.get('bn_use',True)
        group_num = kargs.get('group_num',32)
        eps = kargs.get('eps',1e-05)
        relu_type = kargs.get('relu_type',None)
        #with tf.variable_scope(name_scope):
        bn_out = tfc.conv2d(data_in,filter_num,kernel_size,conv_stride,activation_fn=None,\
                            trainable=train_fg,weights_regularizer=w_regular,scope='%s_conv' % name_scope)
        #bn_out = tfc.group_norm(conv_out,group_num,epsilon=eps,scope='%s_bn' % name_scope)
        if bn_use:
            bn_out = tfc.layer_norm(bn_out,scope='%s_bn' % name_scope)
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

def res_block(data_in,**kargs):
        filter_num_inside = kargs.get('filter_num_inside',32)
        filter_num_out = kargs.get('filter_num_out',64)
        shape_same = kargs.get('shape_same',True)
        name_scope = kargs.get('res_name','res1_1')
        r_relu = kargs.get('rrelu_type','relu6')
        res_stride = kargs.get('res_stride',1)
        #with tf.variable_scope(name_scope):
        res1a_1 = Conv_block(data_in,1,filter_num=filter_num_inside,relu_type=r_relu,\
                            conv_stride=res_stride,name='%sa_1' % name_scope,**kargs)
        res1a_2 = Conv_block(res1a_1,3,filter_num=filter_num_inside,relu_type=r_relu,\
                            name='%sa_2' % name_scope,**kargs)
        res1a_3 = Conv_block(res1a_2,1,filter_num=filter_num_out,\
                            name='%sa_3' % name_scope,**kargs)
        if shape_same:
            short_cut = data_in
        else:
            short_cut = Conv_block(data_in,1,filter_num=filter_num_out,conv_stride=res_stride,\
                        name='%sb' % name_scope,**kargs)
        res_out = tf.add(short_cut,res1a_3,name='%s_add' % name_scope)
        if r_relu == 'leaky_relu':
            res_out = tf.nn.leaky_relu(res_out,name='%s_prelu' % name_scope)
        elif r_relu == 'relu6':
            res_out = tf.nn.relu6(res_out,name='%s_relu6' % name_scope)
        else :
            res_out = tf.nn.relu(res_out,name='%s_relu' % name_scope)
        return res_out

def res_block_seq(data_in,block_num,**kargs):
        seq_name = kargs.get('seq_name','res1')
        seq_stride = kargs.get('seq_stride',1)
        seq_num_in = kargs.get('kernel_num_in',32)
        seq_num_out = kargs.get('kernel_num_out',64)
        #with tf.variable_scope(seq_name):
        res_seq_out = res_block(data_in,shape_same=False,res_stride=seq_stride,filter_num_inside=seq_num_in,\
                        filter_num_out=seq_num_out,res_name='%s_1' % seq_name,**kargs)
        if block_num >1:
            for idx in range(1,block_num):
                seq_child = seq_name + "_%d" % (idx+1)
                res_seq_out = res_block(res_seq_out,filter_num_inside=seq_num_in,\
                                        filter_num_out=seq_num_out,res_name=seq_child,**kargs)
        return res_seq_out

def get_symble(input_image,**kargs):
        w_decay = kargs.get('w_decay',1e-5)
        net_name = kargs.get('net_name','resnet50')
        train_fg = kargs.get('train_fg',True)
        class_num = kargs.get('class_num',81)
        w_r = tfc.l2_regularizer(w_decay)
        assert net_name.lower() in ['resnet50','resnet100'], "Please sel netname: resnet50 or resnet100"
        with tf.variable_scope(net_name):
            res_base_conv = Conv_block(input_image,7,conv_stride=2,filter_num=64,relu_type='relu6',\
                                    w_regular=w_r,**kargs)
            C1 = tfc.max_pool2d(res_base_conv,3,stride=2,padding='SAME',scope='res_base_pool')
            C2 = res_block_seq(C1,3,kernel_num_in=64,kernel_num_out=256,w_regular=w_r,seq_name='res2',**kargs)
            C3 = res_block_seq(C2,4,seq_stride=2,kernel_num_in=128,kernel_num_out=512,w_regular=w_r,\
                                seq_name='res3',**kargs)
            if 'resnet50' in net_name:
                C4 = res_block_seq(C3,6,seq_stride=2,kernel_num_in=256,kernel_num_out=1024,w_regular=w_r,\
                                seq_name='res4',**kargs)
            elif 'resnet100' in net_name:
                C4 = res_block_seq(C3,23,seq_stride=2,kernel_num_in=256,kernel_num_out=1024,w_regular=w_r,\
                                seq_name='res4',**kargs)
            else:
                print("Please input net name in:[resnet50,resnet100]")
                return None
            C5 = res_block_seq(C4,3,seq_stride=2,kernel_num_in=512,kernel_num_out=2048,w_regular=w_r,\
                                seq_name='res5',**kargs)
            p2 = tfc.avg_pool2d(C5,7,stride=1,padding='SAME',scope='pool2')
            flat = tfc.flatten(p2,scope='flat')
            fc = tfc.fully_connected(flat,class_num,activation_fn=tf.nn.relu6,trainable=train_fg,\
                                        weights_regularizer=w_r,scope='fc')
            dp = tfc.dropout(fc,keep_prob=0.5,is_training=train_fg,scope='drop_out')
            return dp

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        img = tf.ones([64,224,224,3])
        model = get_symble(img,class_num=3,net_name='resnet50')
        sess = tf.Session()
        summary = tf.summary.FileWriter('/home/lxy/Develop/Center_Loss/git_prj/face-anti-spoofing/logs/',sess.graph)
