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


def conv2d(x,W,b,strides=1):
    # conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    # x = tf.nn.conv3d(x,W,strides=[1,strides,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    # max2d wrapper
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(X,weights,biases,dropout):
    #X = tf.reshape(X, shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
    # convolustion layer
    conv1 = conv2d(X,weights['wc1'],biases['bc1'])
    # max pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # convolustion layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # max pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # apply dropout
    # conv2 = tf.nn.dropout(conv2, 0.98)

    # convolustion layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # max pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    
    # apply dropout
    # conv3 = tf.nn.dropout(conv3, 0.95)

    # convolustion layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # max pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)
    
    # apply dropout
    # conv4 = tf.nn.dropout(conv4, 0.9)

    # convolustion layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # max pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)
    
    # apply dropout
    conv5 = tf.nn.dropout(conv5, 0.9)

    # print(conv4.shape)
    # fully connected layer
    fc1 = tf.reshape(conv5, shape=[-1,weights['wd1'].get_shape().as_list()[0]])
    # print('conv4 shape:', conv4.shape, ', fc1 shape:', fc1.shape)
    fc1 = tf.add(tf.matmul(fc1,weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # output, class prediction
    out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])
    return  out

def get_symble_(input_data,class_num):
    # store layers weighta and bias
    weights = {
        # 5x5 conv, 3 inputs, 16 outpus
        'wc1': tf.get_variable('wc1',[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        # 5x5 conv, 16 input, 32 outpus
        'wc2': tf.get_variable('wc2',[3,3,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc3': tf.get_variable('wc3',[3,3,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        # 5x5 conv, 64 inputs, 128 outputs
        'wc4': tf.get_variable('wc4',[3,3,128,256],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        # 5x5 conv, 128 inputs, 256 outputs
        'wc5': tf.get_variable('wc5', [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),

        # fully connected, 7*7*128 inputs, 2048 outputs
        'wd1': tf.get_variable('wd1',[1*1*512,2048],initializer=tf.contrib.layers.xavier_initializer()),
        # 32 inputs, 26 outputs (class prediction)
        'out': tf.get_variable('fc1',[2048,class_num],initializer=tf.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.Variable(tf.zeros([32])),
        'bc2': tf.Variable(tf.zeros([64])),
        'bc3': tf.Variable(tf.zeros([128])),
        'bc4': tf.Variable(tf.zeros([256])),
        'bc5': tf.Variable(tf.zeros([512])),
        'bd1': tf.Variable(tf.zeros([2048])),
        'out': tf.Variable(tf.zeros([class_num]))
    }


    # cconstruct model
    logits = conv_net(input_data,weights,biases,0.5)
    #prediction = tf.nn.softmax(logits)
    return logits

if __name__=='__main__':
    graph = tf.Graph()
    with graph.as_default():
        img = tf.ones([64,112,112,3])
        model = get_symble(img,class_num=93,net_name='lenet5',train_fg=True)
        sess = tf.Session()
        summary = tf.summary.FileWriter('/home/lxy/Develop/Center_Loss/git_prj/face-anti-spoofing/logs/',sess.graph)