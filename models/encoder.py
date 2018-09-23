
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

class encoder:
    def __init__(self, imsize=256, nlabels=2, 
        convspec=CONV_DEFAULT, poolspec=POOL_DEFAULT,
        unconvspec=UNCONV_DEFAULT,
        unpoolspec=UNPOOL_DEFAULT):

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
        
        input = Input(shape=(imsize, imsize, 1))
        carry = input

        # skips hold pre-maxpool output layers
        carry, skips = auto_conv(carry, convspec, poolspec)
        assert len(skips) == len(unconvspec)

        fmap = carry # this is the 4x4x512 featuremap
            
        carry = Reshape((dynsize * dynsize * 512,))(carry)
        carry = Dense(2048)(carry)
        carry = BatchNormalization()(carry)
        carry = Activation('relu')(carry)

        carry = Dense(dynsize * dynsize * 512)(carry)
        carry = BatchNormalization()(carry)
        carry = Activation('relu')(carry)

        # back to 2d shape
        carry = Reshape((dynsize, dynsize, 512,))(carry)

        carry = auto_unconv(carry, unconvspec, unpoolspec, skips)
        carry = Conv2D(1, (3, 3), padding='same')(carry)
        carry = Activation('sigmoid')(carry)

        assert carry.get_shape().as_list() == [None, imsize, imsize, 1]

        model = Model(input, carry)
        self.model = model

    def model(self):
        return self.model

    def compile(self):
        def mse(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)
        
        self.core = self.model
        self.model = mgpu(self.model)
        opt = Adam(0.001)
        self.model.compile(optimizer=opt, loss=mse)

