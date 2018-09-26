
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

GAP_CONV = [
    [64, 128],
    [256, 256, 256, 256], # 128
    [256, 256, 256, 256], # 64
    [512, 512, 512, 512],# 32
    [512, 512, 512, 512],# 16
    [512, 512, 512, 512],# 8
    [1024, 1024, 2048, 2048],# 8
] # 4

GAP_POOL = [
    True, True, True, 
    True, True, False,
    False,
    False]

class gap:
    name = 'gap_v2'

    def __init__(self, imsize=256, nlabels=2, 
        convspec=GAP_CONV, poolspec=GAP_POOL):

        print(' [!] Conv Spec:')
        dynsize = imsize
        for lii,layerspec in enumerate(convspec):
            tosize = dynsize//2 if poolspec[lii] else dynsize
            if tosize % 2 == 1: tosize += 1
            print(' [!] %d => %d: %s' % (dynsize, tosize, layerspec))
            dynsize = tosize
        
        input = Input(shape=(imsize, imsize, 1))
        carry = input

        carry, _ = auto_conv(carry, convspec, poolspec)
            
        carry = Reshape((dynsize, dynsize, 2048))(carry)
        featuremaps = carry
        carry = AveragePooling2D((dynsize, dynsize))(carry)
        carry = Reshape((2048,))(carry)
        # carry = Dense(2048)(carry)
        # carry = BatchNormalization()(carry)
        # carry = Activation('relu')(carry)
        avgweights = carry
        carry = Dropout(0.5)(carry)
        # carry = Dropout(0.5)(carry)
        # carry = Dense(2048)(carry)
        # carry = BatchNormalization()(carry)
        # carry = Activation('relu')(carry)
        # carry = Dropout(0.5)(carry)
        carry = Dense(nlabels)(carry)
        carry = Activation('softmax')(carry)

        model = Model(input, carry)

        self.model = model
        self.core = model
        self.saliency_model = Model(input, [carry, featuremaps])

    def summary(self):
        self.model.summary()

    def count_params(self):
        return '%.1f million' % (int(self.model.count_params()) / (1000 * 1000))

    def saliency(self, imgs):
        return self.saliency_model.predict(imgs) 
        # return 

    def save(self):
        self.core.save_weights('checkpoints/%s.h5' % self.name)

    def load(self, path=None):
        if path is not None:
            self.core.load_weights(path)
        else:
            self.core.load_weights('checkpoints/%s.h5' % self.name)

    def compile(self):
        self.core = self.model
        self.model = mgpu(self.model)
        self.model = self.model
        opt = Adam(0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

