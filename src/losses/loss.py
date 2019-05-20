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
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__),"../configs"))
from config import cfgs


def entropy_loss(logits,labels,class_num,lr_mult=1):
    '''
    labels: Tensor of target data from the generator with shape (B, num_classes).
    logits: Tensor of predicted data from the network with shape (B,num_classes).
    '''
    soft_logits=tf.nn.softmax(logits)
    pred = tf.argmax(soft_logits,axis=1)
    labels = tf.cast(tf.reshape(labels, [-1]),tf.int32)
    labels_hot = tf.one_hot(labels,class_num)
    batch_size = logits.get_shape()[0]
    labels_hot = tf.reshape(labels_hot,[batch_size,-1])
    #cross_entropy = tf.losses.softmax_cross_entropy(labels_hot,logits)
    cross_entropy = -tf.reduce_sum(labels_hot * tf.log(soft_logits), axis=1)
    bg = tf.zeros_like(labels,dtype=tf.float32)
    ones_ = tf.ones_like(labels,dtype=tf.float32)
    fg = tf.constant(lr_mult,shape=[batch_size],dtype=tf.float32)
    wd = tf.where(tf.greater(tf.cast(labels,tf.float32),bg),fg,ones_)
    pred_wd = tf.where(tf.equal(labels,tf.cast(pred,tf.int32)),ones_,wd)
    pred_wd_stop = tf.stop_gradient(pred_wd)
    cross_entropy = cross_entropy * pred_wd_stop
    cross_loss = tf.reduce_mean(cross_entropy)
    return cross_loss,soft_logits

def focal_loss(logits,labels,class_num,alpha=0.25,gamma=1):
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

def calc_focal_loss(cls_outputs, cls_targets, alpha=0.25, gamma=2.0):
    """
    Args:
        cls_outputs: [batch_size, num_classes]
        cls_targets: [batch_size, num_classes]
    Returns:
        cls_loss:[1]

    Compute focal loss:
        FL = -(1 - pt)^gamma * log(pt), where pt = p if y == 1 else 1 - p
        cf. https://arxiv.org/pdf/1708.02002.pdf
    """
    cls_targets = tf.cast(tf.reshape(cls_targets, [-1,cfgs.CLS_NUM]),tf.int32)
    positive_mask = tf.equal(cls_targets, 1)
    pos = tf.where(positive_mask, 1.0 - cls_outputs, tf.zeros_like(cls_outputs))
    neg = tf.where(positive_mask, tf.zeros_like(cls_outputs), cls_outputs)
    pos_loss = - alpha * tf.pow(pos, gamma) * tf.log(tf.clip_by_value(cls_outputs, 1e-15, 1.0))
    neg_loss = - (1 - alpha) * tf.pow(neg, gamma) * tf.log(tf.clip_by_value(1.0 - cls_outputs, 1e-15, 1.0))
    loss = tf.reduce_sum(pos_loss + neg_loss, axis=0)
    total_loss = tf.reduce_sum(pos_loss+neg_loss)
    return total_loss,loss

def calc_acc(cls_outputs,cls_targets):
    """
    Args:
        cls_outputs: [batch_size, num_classes]
        cls_targets: [batch_size, num_classes]
    Returns:
        acc: number(0-100)
    """
    pred_mask = tf.greater(cls_outputs,0.5)
    pred_label = tf.where(pred_mask,tf.ones_like(cls_targets),tf.zeros_like(cls_targets))
    pred_label = tf.cast(pred_label,tf.int32)
    cls_targets = tf.cast(cls_targets,tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_label,cls_targets),tf.float32),axis=0)
    return acc,cls_targets,pred_label

if __name__ == '__main__':
    pred = tf.constant([[0.2,0.3],[0.8,0.4],[0.1,0.9]],dtype=tf.float32)
    label = tf.constant([[0,1],[1,0],[0,1]],dtype=tf.int32)
    print(pred.shape,label.shape)
    sess = tf.Session()
    #err = focal_loss(pred,label,2)
    #err = entropy_loss(pred,label,2)
    err = calc_focal_loss(pred,label)
    er_o = sess.run(err)
    print('out',er_o)
    ac = calc_acc(pred,label)
    acc = sess.run(ac)
    print('acc',acc)