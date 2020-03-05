'''
- Author      : Nikita Malviya
- Start Date  : 13/Feb/19
- Last Edited : 10/Apr/19 '''

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Flatten, Input
from keras.layers import MaxPooling2D, AveragePooling2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

input_shape = (48, 48, 1)
num_classes = 7
l2_regularization=0.01
regularization = l2(l2_regularization)

def __init__(input_shape = (48, 48, 1),  num_classes = 7, l2_regularization=0.01):
    # data generator
    data_generator = ImageDataGenerator(
                    featurewise_center=False,
                    featurewise_std_normalization=False,
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=.1,
                    horizontal_flip=True)

    regularization = l2(l2_regularization)

def define_model(input_shape, num_classes, regularization):
    # base
    img_input = Input(input_shape)
    x = Conv2D(filters = 8, kernel_size = (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    # 8 dimensions of input, 8*3*3 filters, (8*3*3+1)*feature maps as output
    # size of output CNN layer = input_shape-(filter_size-1) = 48-2 = 46
    # Convolutional network should not be less than the input, so padding is done.
    # If we want to explicitly want to downsample the image during the convolutional, we can define a stride.
    # To calculate padding, input_size + 2 * padding_size-(filter_size-1) = 48+(2*1)-2 = 48
    # number of parameters the network learned = (n*m*k+1)*f = (filter_size[0]*filter_size[1]*feature_map_as_output+1)*no_of_filters
    # Dropout : removes the nodes that are below the weights mentioned.

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = 8, kernel_size = (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(filters = 16, kernel_size = (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(filters = 16, kernel_size = (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters = 16, kernel_size = (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(filters = 32, kernel_size = (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(filters = 32, kernel_size = (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters = 32, kernel_size = (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(filters = 64, kernel_size = (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(filters = 64, kernel_size = (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters = 64, kernel_size = (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(filters = 128, kernel_size = (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(filters = 128, kernel_size = (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters = 128, kernel_size = (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(filters = num_classes, kernel_size = (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)
    return img_input, output

__init__()
img_input, model_defination = define_model(input_shape, num_classes, regularization)
model = Model(img_input, model_defination)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# summary = model.summary()