# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/17 17:09
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
    with tf.variable_scope(name_scope):
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

def DepConv_block(data_in,kernel_size,**kargs):
    conv_stride = kargs.get('dconv_stride',1)
    name_scope = kargs.get('name','res_base')
    w_regular = kargs.get('w_regular',None)
    train_fg = kargs.get('train_fg',True)
    bn_use = kargs.get('bn_use',True)
    group_num = kargs.get('group_num',32)
    eps = kargs.get('eps',1e-05)
    relu_type = kargs.get('relu_type','relu')
    with tf.variable_scope(name_scope):
        bn_out = tfc.conv2d_in_plane(data_in,kernel_size,conv_stride,activation_fn=None,\
                            trainable=train_fg,weights_regularizer=w_regular,scope='%s_dconv' % name_scope)
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

def Inverted_residual_block(data_in,**kargs):
    filter_num_inside = kargs.get('filter_num_inside',32)
    filter_num_out = kargs.get('filter_num_out',64)
    shape_same = kargs.get('shape_same',True)
    name_scope = kargs.get('res_name','res1_1')
    r_relu = kargs.get('rrelu_type','relu6')
    res_stride = kargs.get('res_stride',1)
    with tf.variable_scope(name_scope):
        res1a_1 = Conv_block(data_in,1,filter_num=filter_num_inside,relu_type=r_relu,\
                            name='%sa_1' % name_scope,**kargs)
        res1a_2 = DepConv_block(res1a_1,3,dconv_stride=res_stride,relu_type=r_relu,\
                            name='%sa_2' % name_scope,**kargs)
        res1a_3 = Conv_block(res1a_2,1,filter_num=filter_num_out,\
                            name='%sa_3' % name_scope,**kargs)
        if res_stride==1 :
            if not shape_same:
                res_out = res1a_3
            else:
                res_out = tf.add(data_in,res1a_3,name='%s_add' % name_scope)
        else:
            res_out = res1a_3
        return res_out

def Inverted_residual_seq(data_in,t,chal_in,chal_out,stride,block_num,**kargs):
    seq_name = kargs.get('seq_name','res1')
    with tf.variable_scope(seq_name):
        res_seq_out = Inverted_residual_block(data_in,shape_same=False,res_stride=stride,filter_num_inside=t*chal_in,\
                        filter_num_out=chal_out,res_name='%s_1' % seq_name,**kargs)
        if block_num >1:
            for idx in range(1,block_num):
                seq_child = seq_name + "_%d" % (idx+1)
                res_seq_out = Inverted_residual_block(res_seq_out,filter_num_inside=t*chal_in,\
                                        filter_num_out=chal_out,res_name=seq_child,**kargs)
        return res_seq_out

def GlobalAveragePooling2D(x,**kargs):
    assert x.get_shape().ndims == 4
    name_scope = kargs.get('name','global_pool')
    return tf.reduce_mean(x, [1, 2],name=name_scope)

def get_symble(input_image,**kargs):
    w_decay = kargs.get('w_decay',1e-5)
    net_name = kargs.get('net_name','mobilenet')
    w_r = tfc.l2_regularizer(w_decay)
    width_mult = kargs.get('width_mult',1.0)
    train_fg = kargs.get('train_fg',True)
    class_num = kargs.get('class_num',81)
    assert net_name.lower() in ['mobilenet','mobilenetv2'],"Please sel netname: mobilenet or mobilenetv2"
    cn = [int(x*width_mult) for x in [32,16,24,32,64,96,160,320,1280]]
    with tf.variable_scope(net_name) :
        b0 = Conv_block(input_image,3,filter_num=cn[0],conv_stride=2,relu_type='relu6', \
                        name='cb1',w_regular=w_r,**kargs)
        b1 = Inverted_residual_seq(b0,1,cn[0],cn[1],1,1,**kargs)
        b2 = Inverted_residual_seq(b1,6,cn[1],cn[2],2,2,seq_name='res2',w_regular=w_r,**kargs)
        b3 = Inverted_residual_seq(b2,6,cn[2],cn[3],2,3,seq_name='res3',w_regular=w_r,**kargs)
        b4 = Inverted_residual_seq(b3,6,cn[3],cn[4],2,4,seq_name='res4',w_regular=w_r,**kargs)
        b5 = Inverted_residual_seq(b4,6,cn[4],cn[5],1,3,seq_name='res5',w_regular=w_r,**kargs)
        b6 = Inverted_residual_seq(b5,6,cn[5],cn[6],2,3,seq_name='res6',w_regular=w_r,**kargs)
        b7 = Inverted_residual_seq(b6,6,cn[6],cn[7],1,1,seq_name='res7',w_regular=w_r,**kargs)
        b8 = Conv_block(b7,1,filter_num=cn[8],conv_stride=1,relu_type='relu6', \
                        name='cb2',**kargs)
        pool = GlobalAveragePooling2D(b8,name='pool')
        fc = tfc.fully_connected(pool,class_num,activation_fn=tf.nn.relu6,trainable=train_fg,\
                                    weights_regularizer=w_r,scope='fc')
        return fc

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        img = tf.ones([1,224,224,3])
        model = get_symble(img,class_num=2,net_name='mobilenet')
        sess = tf.Session()
        summary = tf.summary.FileWriter('/home/lxy/Develop/Center_Loss/git_prj/face-anti-spoofing/logs/',sess.graph)
        #out = sess.run(model)
        #print(np.shape(out))