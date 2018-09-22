

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

POOL_DEFAULT = [True] * 6

CONV_DEFAULT = [
    [64, 128],
    [256, 256, 256, 256], # 128
    [256, 256, 256, 256], # 64
    [256, 256, 256, 256], # 64
#     [256, 256, 256], # 32
    [512, 512, 512, 512],# 16
    [512, 512, 512, 512],# 16
#     [512, 512, 512], # 8
] # 4

def auto_conv(carry, convspec, poolspec):
    skips = []
    for lii, layerspec in enumerate(convspec):
        for fii, nFilters in enumerate(layerspec):
            carry = Conv2D(nFilters, (3, 3), padding='same')(carry)
            carry = BatchNormalization()(carry)
            if fii + 1 == len(layerspec): skips.append(carry) # save last
            carry = Activation('relu')(carry)
        if poolspec[lii]:
            carry = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(carry)
    return carry, skips