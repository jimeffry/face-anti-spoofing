# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/19 17:09
#project: face anti spoofing
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face anti spoofing
####################################################
import numpy as np 
import tensorflow as tf 


def entropy_loss(logits,labels,class_num):
    '''
    labels: Tensor of target data from the generator with shape (B, num_classes).
    logits: Tensor of predicted data from the network with shape (B,num_classes).
    '''
    soft_logits=tf.nn.softmax(logits)
    labels = tf.cast(tf.reshape(labels, [-1]),tf.int32)
    labels_hot = tf.one_hot(labels,class_num)
    batch_size = logits.get_shape()[0]
    labels_hot = tf.reshape(labels_hot,[batch_size,-1])
    #cross_entropy = tf.losses.softmax_cross_entropy(labels_hot,logits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_hot)
    #cross_entropy = -tf.reduce_sum(labels_hot * tf.log(soft_logits), axis=1)
    #cross_entropy = -labels_hot * tf.log(logits)
    cross_loss = tf.reduce_mean(cross_entropy)
    return cross_loss,soft_logits

def focal_loss(logits,labels,class_num,alpha=0.25,gamma=2):
    """ 
    Compute the focal loss given the target tensor and the predicted tensor.
    As defined in https://arxiv.org/abs/1708.02002
    args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    # compute the focal loss
    softmax_out=tf.nn.softmax(logits)
    labels = tf.cast(tf.reshape(labels, [-1]),tf.int32)
    labels_hot = tf.one_hot(labels,class_num)
    #softmax_out = tf.constant(logits,shape=logits.shape,dtype=tf.float32)
    batch_size = logits.get_shape()[0]
    #batch_size = logits.shape[0]
    alpha = tf.constant(alpha,shape=[batch_size],dtype=tf.float32)
    gamma = tf.constant(gamma,shape=[batch_size],dtype=tf.float32)
    alpha_factor = tf.multiply(tf.ones_like(labels,dtype=tf.float32),alpha)
    #bg = tf.zeros_like(labels)
    #alpha_factor = tf.where(tf.greater(labels, bg), alpha_factor, 1 - alpha_factor)
    #focal_weight = tf.where(tf.greater(labels, bg), 1 - softmax_out, softmax_out)
    h_idx = tf.range(batch_size,dtype=tf.int32)
    sel = tf.stack([h_idx,labels])
    sel = tf.transpose(sel)
    sel_soft = tf.gather_nd(softmax_out,sel)
    focal_weight = 1 - sel_soft
    #print('w',focal_weight.get_shape())
    focal_weight=tf.pow(focal_weight, gamma)
    focal_weight = tf.multiply(alpha_factor, focal_weight)
    cross_entropy = -tf.reduce_sum(labels_hot * tf.log(softmax_out),axis=1)
    #cls_loss = tf.reduce_sum(focal_weight * cross_entropy,axis=0) / tf.cast(batch_size,tf.float32)
    cls_loss = tf.reduce_mean(focal_weight * cross_entropy)
    return cls_loss,softmax_out

def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(tf.reshape(label, [-1]),tf.int64)
    #cond = tf.where(tf.greater_equal(label_int,0))
    #picked = tf.squeeze(cond)
    #label_picked = tf.gather(label_int,picked)
    #pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_int,pred),tf.float32))
    return accuracy_op,label_int,pred,cls_prob

if __name__ == '__main__':
    pred = tf.constant([[0.2,0.3],[0.8,0.4],[0.1,0.9]],dtype=tf.float32)
    label = np.array([1,1,1])
    print(pred.shape,label.shape)
    sess = tf.Session()
    #err = focal_loss(pred,label,2)
    err = entropy_loss(pred,label,2)
    er_o = sess.run(err)
    print('out',er_o[0])
    ac = cal_accuracy(er_o[1],label)
    acc = sess.run(ac)
    print('acc',acc)