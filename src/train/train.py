# -*- coding:utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/12/10 15:09
#project: Face detect
#company: 
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  face detect 
####################################################
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib as tfc
import os, sys
import numpy as np
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
import mobilenetV2
import resnet
sys.path.append(os.path.join(os.path.dirname(__file__),"../prepare_data"))
from read_tfrecord import Read_Tfrecord
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from get_property import load_property
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from loss import focal_loss,cal_accuracy

def parms():
    parser = argparse.ArgumentParser(description='SSH training')
    parser.add_argument('--load-num',dest='load_num',type=int,default=0,help='ckpt num')
    parser.add_argument('--save-weight-period',dest='save_weight_period',type=int,default=5,\
                        help='the period to save')
    parser.add_argument('--epochs',type=int,default=20000,help='train epoch nums')
    parser.add_argument('--batch-size',dest='batch_size',type=int,default=32,\
                        help='train batch size')
    parser.add_argument('--model-path',dest='model_path',type=str,default='../../models/ssh',\
                        help='path saved models')
    parser.add_argument('--log-path',dest='log_path',type=str,default='../../logs',\
                        help='path saved logs')
    parser.add_argument('--gpu-list',dest='gpu_list',type=str,default='0',\
                        help='train on gpu num')
    parser.add_argument('--property-file',dest='property_file',type=str,\
                        default='../../data/property.txt',help='nums of train dataset images')
    parser.add_argument('--data-record-dir',dest='data_record_dir',type=str,\
                        default='../../data/',help='tensorflow data record')
    return parser.parse_args()

def train(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    load_num = args.load_num
    log_dir = args.log_path
    epochs = args.epochs
    batch_size = args.batch_size
    save_weight_period = args.save_weight_period
    data_record_dir = args.data_record_dir
    property_file = os.path.join(data_record_dir,cfgs.DATASET_NAME,'property.txt')
    Property = load_property(property_file)
    train_img_nums = Property['img_nums']
    class_nums = Property['cls_num']
    with tf.variable_scope('get_batch'):
        tfrd = Read_Tfrecord(cfgs.DATASET_NAME,data_record_dir,batch_size,True)
        img_name_batch, img_batch, label_batch = tfrd.next_batch()
        #gtboxes_and_label = tf.reshape(label_batch, [-1])
    # list as many types of layers as possible, even if they are not used now
    with tf.variable_scope('build_trainnet'):
        if cfgs.NET_NAME in 'mobilenetv2':
            logits = mobilenetV2.get_symbol(img_batch,w_decay=cfgs.WEIGHT_DECAY,\
                                        class_num=class_nums,train_fg=True)
        elif cfgs.NET_NAME in ['resnet50','resnet100']:
            logits = resnet.get_symbol(img_batch,w_decay=cfgs.WEIGHT_DECAY,\
                                        class_num=class_nums,train_fg=True)
    # ----------------------------------------------------------------------------------------------------build loss
    with tf.variable_scope('build_loss'):
        weight_decay_loss = tf.add_n(tf.losses.get_regularization_losses())
        cls_loss,soft_logits = focal_loss(logits,label_batch,class_nums)
        total_loss = cls_loss + weight_decay_loss
        acc_op = cal_accuracy(soft_logits,label_batch)
    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('LOSS/cls_loss', cls_loss)
    tf.summary.scalar('LOSS/total_loss', total_loss)
    tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)
    # ---------------------------------------------------------------------------------------------------learning rate
    global_step = tf.train.create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(x) for x in cfgs.DECAY_STEP],
                                     values=[y for y in cfgs.LR])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
    # ---------------------------------------------------------------------------------------------compute gradients
    gradients = optimizer.compute_gradients(total_loss)
    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    saver = tf.train.Saver(max_to_keep=30)
    tf_config = tf.ConfigProto()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #tf_config.gpu_options = gpu_options
    tf_config.gpu_options.allow_growth=True  
    tf_config.log_device_placement=False
    with tf.Session(config=tf_config) as sess:
        sess.run(init_op)
        if load_num >0:
            model_path = "%s-%s" %(model_path,str(load_num) )
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print("restore model path:",model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            saver.restore(sess, model_path)
            print("restore models' param")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_path = os.path.join(log_dir,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
        try:
            for epoch_tmp in range(epochs):
                for step in range(np.ceil(train_img_nums/batch_size).astype(np.int32)):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                        _, global_stepnp = sess.run([train_op, global_step])
                    else:
                        if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                            start = time.time()
                            _, global_stepnp, totalLoss,cls_l,acc = sess.run([train_op, global_step, total_loss,cls_loss,acc_op])
                            end = time.time()
                            print(""" %s epoch:%d step:%d | per_cost_time:%.3f s | total_loss:%.3f | cls_loss:%.3f | acc:%.4f """ \
                                % (str(training_time), epoch_tmp,global_stepnp, (end - start),totalLoss,cls_l,acc))
                        else:
                            if step % cfgs.SMRY_ITER == 0:
                                _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                                summary_writer.add_summary(summary_str, global_stepnp)
                                summary_writer.flush()
                if (epoch_tmp > 0 and epoch_tmp % save_weight_period == 0) or (epoch_tmp == epochs - 1):
                    save_dir = model_path
                    saver.save(sess, save_dir,epoch_tmp)
                    print(' weights had been saved')
        except tf.errors.OutOfRangeError:
            print("Trianing is over")
        finally:
            coord.request_stop()
            summary_writer.close()
        coord.join(threads)
        #record_file_out.close()
        sess.close()

if __name__ == '__main__':
    args = parms()
    gpu_group = args.gpu_list
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    train(args)




