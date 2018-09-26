#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shielding
"""

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500 
REGULARIZER = 0.0001

def get_weight(shape, regularize):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1)) #参数满足截断正态分布
    if REGULARIZER!=None: 
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZER)(w)) # 正则化
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape)) # 初始bias置0
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE],regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    
    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE],regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1,w2) + b2 #softmax函数使结果符合概率分布
    return y
    
