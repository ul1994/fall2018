
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

class yprob:
    def __init__(self, imsize=256, nlabels=2):
        print(' [!] Conv Spec:')
        dynsize = imsize
        for lii,layerspec in enumerate(convspec):
            tosize = dynsize//2 if poolspec[lii] else dynsize
            if tosize % 2 == 1: tosize += 1
            print(' [!] %d => %d: %s' % (dynsize, tosize, layerspec))
            dynsize = tosize
        
        model = Sequential()
        # NOTE: Initial layers needs an input_shape...
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', 
                        input_shape=(imsize, imsize, 1)))

        skips = auto_conv(model)
            
        model.add(Reshape((dynsize * dynsize * 512,)))
        model.add(Dense(2048))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nlabels))
        model.add(Activation('softmax'))

        self.model = model

    def compile(self):
        self.model = mgpu(self.model)
        opt = Adam(0.001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

