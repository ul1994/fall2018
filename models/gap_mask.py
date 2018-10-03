
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
    [512, 512, 512, 512],# 16
    [1024, 1024, 2048, 2048],# 8
] # 4

GAP_POOL = [
    True, True, True,
    True, True, False,
    False,
    False]

class gap_mask:
    name = 'gap_mask_v1'

    def __init__(self,
        imsize=256,
        nlabels=2,
        masksize=32,
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
        fmap = carry

        # rpnbase
        carry = Conv2D(2048, (3, 3), padding='same')(carry)
        carry = BatchNormalization()(carry)
        carry = Activation('relu')(carry)
        rpnbase = carry

        carry = Conv2D(512, (3, 3), padding='same')(carry)
        carry = BatchNormalization()(carry)
        carry = Activation('relu')(carry)

        # rpn logits
        carry = Conv2D(2, (1, 1), padding='same')(carry)
        rpn = Activation('softmax', name='rpn')(carry)

        # print(carry.get_shape())


        carry = AveragePooling2D((dynsize, dynsize))(fmap)
        carry = Reshape((2048,))(carry)
        coding = carry
        carry = Dropout(0.5)(carry)
        carry = Dense(nlabels)(carry)
        yhat = Activation('softmax', name='yhat')(carry)

        model = Model(input, [rpn, yhat])
        self.model = model
        self.core = model
        self.coding = Model(input, coding)
        # self.saliency_model = Model(input, [featuremaps, avgweights])

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
        self.core = self.model
        self.model = mgpu(self.model)
        self.model = self.model
        opt = Adam(0.0001)

        def mse(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true), axis=-1)
        def lossy_iou(ytrue, yguess, oob=1, iib=1):
            # any prediction outside boundary should be identical
            outside_penalty = oob * mse(
                # compared against mask (treated as loose bound)
                ytrue[:, :, :, 0],
                # mask out potential bg prediction within bounds
                #  this only works b/c assumption that ytrue is outer bound
                ytrue[:, :, :, 0] * yguess[:, :, :, 0])

            # print((ytrue[:, :, :, 1] * yguess[:, :, :, 1]))

            # discount for any prediction within mask
            #  no mse possible here because exact boundary is not known
            inside_discount = iib * K.mean(
                K.square(
                    ytrue[:, :, :, 1] * yguess[:, :, :, 1]), axis=-1)

            return outside_penalty - inside_discount

        def outer_penalty(ytrue, yguess):
            return mse(
                # compared against mask (treated as loose bound)
                ytrue[:, :, :, 0],
                # mask out potential bg prediction within bounds
                #  this only works b/c assumption that ytrue is outer bound
                ytrue[:, :, :, 0] * yguess[:, :, :, 0])

        self.model.compile(
            optimizer=opt,
            loss={
                'rpn': lossy_iou,
                'yhat': 'categorical_crossentropy',
            },
            metrics={
                'yhat': 'accuracy',
                'rpn': outer_penalty,
            })

