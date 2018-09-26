
from configs import *
from .common import *

import os, sys
import cv2
import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.layers import Input, Reshape, Dense, Conv2D, Dropout, \
    MaxPooling2D, Flatten, UpSampling2D, Multiply, Activation, AveragePooling2D, \
    Add, Subtract, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.utils import multi_gpu_model as mgpu
from keras.utils import np_utils

h_conv = [
    [64, 128],
    [256, 256, 256], # 256
    [256, 256, 256], # 128
    [256, 256, 256], # 64
    [512, 512, 512],# 32
    [512, 512, 512],# 16
    [512, 512, 512],# 8
] # 4
h_pool = [True] * 7

h_unconv = [ # 4
    [512//2, 512, 512, 512],# 
    [512//2, 512, 512, 512],# 
    [512//2, 512, 512, 512],# 16
    [256//2, 256, 256, 256], # 128
    [256//2, 256, 256, 256], # 64
    [256//2, 256, 256, 256], # 64
    [128//2, 128, 64],
] # 512
h_unpool = [True] * 7


filter_spec = [
    (512, 7),
    (128, 7),
    (128, 7),
    (128, 7),
    (128, 7),
    (128, 1),
    (1, 1),
]

class hcoder:
    name = 'hcoder'
    def __init__(self, imsize=512, hsize=64,
        convspec=h_conv, 
        poolspec=h_pool,
        unconvspec=h_unconv,
        unpoolspec=h_unpool):

        print(' [!] Conv Spec:')
        dynsize = imsize
        for lii,layerspec in enumerate(convspec):
            tosize = dynsize//2 if poolspec[lii] else dynsize
            if tosize % 2 == 1: tosize += 1
            print('  |  %d => %d: %s' % (dynsize, tosize, layerspec))
            dynsize = tosize

        print(' [!] Unconv Spec:')
        dynout = dynsize
        for lii,layerspec in enumerate(unconvspec):
            insize = dynout
            dynout *= 2
            print('  |  %d => %d: %s' % (insize, dynout, layerspec))
        
        iminput = Input(shape=(imsize, imsize, 1))
        # hinput = Input(shape=(hsize, hsize, 1))
        carry = iminput

        # skips hold pre-maxpool output layers
        carry, skips = auto_conv(carry, convspec, poolspec)
        assert len(skips) == len(unconvspec)

        carry = Dropout(0.5)(carry)

        carry = auto_unconv(carry, unconvspec, unpoolspec, None)
        carry = Conv2D(1, (3, 3), padding='same')(carry)
        carry = Activation('sigmoid')(carry)

        assert carry.get_shape().as_list() == [None, imsize, imsize, 1]

        model = Model(iminput, carry)
        self.model = model

    def summary(self):
        self.model.summary()

    def count_params(self):
        return '%.1f million' % (int(self.model.count_params()) / (1000 * 1000))

    def save(self):
        self.core.save_weights('checkpoints/%s.h5' % self.name)

    def load(self, path=None):
        if path is not None:
            self.core.load_weights(path)
        else:
            self.core.load_weights('checkpoints/%s.h5' % self.name)

    def compile(self):
        def mse(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)
        
        self.core = self.model
        self.model = mgpu(self.model)
        opt = Adam(0.001)
        self.model.compile(optimizer=opt, loss=mse)

