from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,MaxPooling3D, UpSampling2D, Dropout, Bidirectional, BatchNormalization, add
from keras.layers import concatenate, Conv2DTranspose,Conv3DTranspose
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras_contrib.layers import CRF

from skimage.transform import rotate, resize
from skimage import data
from skimage import exposure

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from losses import *

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def get_unet_3block(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5):
    print ("no get_unet_3block")
    return 0

def get_unet_3block_no_bn(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5):
    print ("no get_unet_3block_no_bn")
    return 0


def get_unet_3block_single_conv(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5):
    print ("no get_unet_3block_single_conv")
    return 0

def get_unet_4block(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5,dc=1, UPSAMPLE_MODE='DECONV'):
    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
        print ("deconv")
    elif UPSAMPLE_MODE=='SIMPLE':
        upsample=upsample_simple

    inputs = Input((img_size, img_size, 1))
    print ("4blocks unet")

    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    # Middle
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(pool4)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)

    # 8 -> 16
    #deconv4 = Conv2DTranspose(start_neurons * 8, k_size, strides=(2, 2), padding="same")(convm)
    deconv4 = upsample(start_neurons * 8, k_size, strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)

    # 16 -> 32
    #deconv3 = Conv2DTranspose(start_neurons * 4, k_size, strides=(2, 2), padding="same")(uconv4)
    deconv3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 32 -> 64
    #deconv2 = Conv2DTranspose(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    deconv2 = upsample(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    conv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    #deconv1 = Conv2DTranspose(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    deconv1 = upsample(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="relu")(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)


    if dc==1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    elif dc==0:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[mean_iou])
    return model

def get_unet_4block_output_conv(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5,dc=1):
    inputs = Input((img_size, img_size, 1))
    print ("4blocks unet")

    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    # Middle
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(pool4)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, k_size, strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, k_size, strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    conv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons//2,k_size, padding="same", activation="relu")(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="relu")(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)


    if dc==1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    elif dc==0:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[mean_iou])
    return model


def get_unet_4block_dilated(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5,dc=1):
    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
        print ("deconv")
    elif UPSAMPLE_MODE=='SIMPLE':
        print ("simple")
        upsample=upsample_simple

    inputs = Input((img_size, img_size, 1))
    print ("4blocks unet")

    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)


    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(8, 8))(conv4)


    # Middle
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(pool4)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)

    # 8 -> 16
    #deconv4 = Conv2DTranspose(start_neurons * 8, k_size, strides=(8, 8), padding="same")(convm)
    deconv4 = upsample(start_neurons * 8, k_size, strides=(8, 8), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)

    # 16 -> 32
    #deconv3 = Conv2DTranspose(start_neurons * 4, k_size, strides=(4, 4), padding="same")(uconv4)
    deconv3 = upsample(start_neurons * 4, k_size, strides=(4, 4), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 32 -> 64
    #deconv2 = Conv2DTranspose(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    deconv2 = upsample(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    conv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    #deconv1 = Conv2DTranspose(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    deconv1 = upsample(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="relu")(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)


    if dc==1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    elif dc==0:
        #model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[mean_iou])
        model.compile(optimizer=Adam(learning_rate, decay=1e-6),
                        loss=dice_p_bce,
                        metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
    return model


def get_unet_5block(img_size=512, start_neurons = 32, k_size=(3,3),learning_rate=1e-5):
    inputs = Input((img_size, img_size, 1))
    print ("5blocks unet")

    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)


    # Middle
    convm = Conv2D(start_neurons * 32, k_size, activation="relu", padding="same")(pool5)
    convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 32, k_size, activation="relu", padding="same")(convm)
    convm = BatchNormalization()(convm)

    deconv5 = Conv2DTranspose(start_neurons * 16, k_size, strides=(2, 2), padding="same")(convm)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(uconv5)
    uconv5 = BatchNormalization()(uconv5)
    uconv5 = Conv2D(start_neurons * 16, k_size, activation="relu", padding="same")(uconv5)
    uconv5 = BatchNormalization()(uconv5)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, k_size, strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(start_neurons * 8, k_size, activation="relu", padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, k_size, strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(start_neurons * 4, k_size, activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, k_size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    conv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, k_size, activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, k_size, strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 1, k_size, activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss=IoU_loss, metrics=[dice_coef, IoU])
    return model


    def load_data(self,nb_rows,nb_cols,image_path,mask_path):
        imgs, msks = tiff.imread(image_path+'/train-volume.tif'), tiff.imread(mask_path+'/train-labels.tif') / 255
        montage_imgs = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.float32)
        montage_msks = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.int8)
        idxs = np.arange(imgs.shape[0])
        np.random.shuffle(idxs)
        idxs = iter(idxs)
        for y0 in range(0, montage_imgs.shape[0], imgs.shape[1]):
            for x0 in range(0, montage_imgs.shape[1], imgs.shape[2]):
                y1, x1 = y0 + imgs.shape[1], x0 + imgs.shape[2]
                idx = next(idxs)
                montage_imgs[y0:y1, x0:x1] = imgs[idx]
                montage_msks[y0:y1, x0:x1] = msks[idx]
        return montage_imgs, montage_msks

# def compile(self,addition,classes,dilate,dilate_rate,loss=bce_dice_loss):
def get_sdunet(img_size=512, start_neurons = 44, k_size=3,learning_rate=1e-5,dc=1, UPSAMPLE_MODE='DECONV'):
    addition = 1
    dilate = 1
    dilate_rate = 2
    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
        print ("deconv")
    elif UPSAMPLE_MODE=='SIMPLE':
        upsample=upsample_simple

    inputs = Input((img_size, img_size, 1))

    down1 = Conv2D(start_neurons, k_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    b1 = BatchNormalization()(down1)
    #b1 = Dropout(rate=0.3)(b1)
    down1 = Conv2D(start_neurons, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b1)
    b2 = BatchNormalization()(down1)
    #b2 = Dropout(rate=0.3)(b2)
    down1pool = MaxPooling2D((2, 2), strides=(2, 2))(b2)
    #down1pool = Dropout(rate=0.3)(down1pool)

    down2 = Conv2D(start_neurons*2, k_size, activation='relu', padding='same', kernel_initializer='he_normal')(down1pool)
    b3 = BatchNormalization()(down2)
    #b3 = Dropout(rate=0.3)(b3)
    down2 = Conv2D(start_neurons*2, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b3)
    b4 = BatchNormalization()(down2)
    #b4 = Dropout(rate=0.3)(b4)
    down2pool = MaxPooling2D((2,2), strides=(2, 2))(b4)
    #down2pool = Dropout(rate=0.3)(down2pool)

    down3 = Conv2D(start_neurons*4, k_size, activation='relu', padding='same', kernel_initializer='he_normal')(down2pool)
    b5 = BatchNormalization()(down3)
    #b5 = Dropout(rate=0.3)(b5)
    down3 = Conv2D(start_neurons*4, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b5)
    b6 = BatchNormalization()(down3)
    #b6 = Dropout(rate=0.3)(b6)
    down3pool = MaxPooling2D((2, 2), strides=(2, 2))(b6)
    #down3pool = Dropout(rate=0.3)(down3pool)

    if dilate == 1:
    # stacked dilated convolution at the bottleneck
        dilate1 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=1, kernel_initializer='he_normal')(down3pool)
        b7 = BatchNormalization()(dilate1)
        #b7 = Dropout(rate=0.3)(b7)
        dilate2 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(b7)
        b8 = BatchNormalization()(dilate2)
        #b8 = Dropout(rate=0.3)(b8)
        dilate3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(b8)
        b9 = BatchNormalization()(dilate3)
        #b9 = Dropout(rate=0.3)(b9)
        dilate4 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b9)
        b10 = BatchNormalization()(dilate4)
        #b10 = Dropout(rate=0.3)(b10)
        dilate5 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b10)
        b11 = BatchNormalization()(dilate5)
        #b11 = Dropout(rate=0.3)(b11)
        dilate6 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', dilation_rate=32, kernel_initializer='he_normal')(b11)
        if addition == 1:
            dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
            #up3 = upsample((2, 2))(dilate_all_added)
            up3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(dilate_all_added)
        else:
            #up3 = upsample((2, 2))(dilate6)
            up3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(dilate6)
    else:
        dilate1 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(down3pool)
        b7 = BatchNormalization()(dilate1)
        #b7 = Dropout(rate=0.3)(b7)
        dilate2 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b7)
        b8 = BatchNormalization()(dilate2)
        #b8 = Dropout(rate=0.3)(b8)
        dilate3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b8)
        b9 = BatchNormalization()(dilate3)
        #b9 = Dropout(rate=0.3)(b9)
        dilate4 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b9)
        b10 = BatchNormalization()(dilate4)
        #b10 = Dropout(rate=0.3)(b10)
        dilate5 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b10)
        b11 = BatchNormalization()(dilate5)
        #b11 = Dropout(rate=0.3)(b11)
        dilate6 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b11)
        if addition ==1:
            dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
            #up3 = upsample((2, 2))(dilate_all_added)
            up3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(dilate_all_added)
        else:
            #up3 = upsample((2, 2))(dilate6)
            up3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(dilate6)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
    up3 = concatenate([down3, up3])
    b12 = BatchNormalization()(up3)
    #b12 = Dropout(rate=0.3)(b12)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b12)
    b13 = BatchNormalization()(up3)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b13)

    #up2 = upsample((2, 2))(up3)
    up2 = upsample(start_neurons * 2, k_size, strides=(2, 2), padding="same")(up3)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    up2 = concatenate([down2, up2])
    b14 = BatchNormalization()(up2)
    #b14 = Dropout(rate=0.3)(b14)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b14)
    b15 = BatchNormalization()(up2)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b15)

    #up1 = upsample((2, 2))(up2)
    up1 = upsample(start_neurons, k_size, strides=(2, 2), padding="same")(up2)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    up1 = concatenate([down1, up1])
    b16 = BatchNormalization()(up1)
    #b16 = Dropout(rate=0.3)(b16)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b16)
    b17 = BatchNormalization()(up1)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b17)
    b18 = BatchNormalization()(up1)

    output_layer = Conv2D(1, 1, activation='relu')(b18)

    model = Model(input=inputs, output=output_layer)
    if dc==1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    elif dc==0:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[mean_iou])
    return model



def get_4sdunet(img_size=512, start_neurons = 32, k_size=3,learning_rate=1e-5,dc=1, UPSAMPLE_MODE='DECONV'):
    addition = 1
    dilate = 1
    dilate_rate = 2

    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
        print ("deconv")
    elif UPSAMPLE_MODE=='SIMPLE':
        upsample=upsample_simple

    inputs = Input((img_size, img_size, 1))

    down1 = Conv2D(start_neurons, k_size, activation='relu', padding='same', dilation_rate=dilate_rate,kernel_initializer='he_normal')(inputs)
    b1 = BatchNormalization()(down1)
    #b1 = Dropout(rate=0.3)(b1)
    down1 = Conv2D(start_neurons, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b1)
    b2 = BatchNormalization()(down1)
    #b2 = Dropout(rate=0.3)(b2)
    down1pool = MaxPooling2D((2, 2), strides=(2, 2))(b2)
    #down1pool = Dropout(rate=0.3)(down1pool)

    down2 = Conv2D(start_neurons*2, k_size, activation='relu', padding='same', dilation_rate=dilate_rate,kernel_initializer='he_normal')(down1pool)
    b3 = BatchNormalization()(down2)
    #b3 = Dropout(rate=0.3)(b3)
    down2 = Conv2D(start_neurons*2, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b3)
    b4 = BatchNormalization()(down2)
    #b4 = Dropout(rate=0.3)(b4)
    down2pool = MaxPooling2D((2,2), strides=(2, 2))(b4)
    #down2pool = Dropout(rate=0.3)(down2pool)

    down3 = Conv2D(start_neurons*4, k_size, activation='relu', padding='same', dilation_rate=dilate_rate,kernel_initializer='he_normal')(down2pool)
    b5 = BatchNormalization()(down3)
    #b3 = Dropout(rate=0.3)(b3)
    down3 = Conv2D(start_neurons*4, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b5)
    b6 = BatchNormalization()(down3)
    #b4 = Dropout(rate=0.3)(b4)
    down3pool = MaxPooling2D((2,2), strides=(2, 2))(b6)
    #down2pool = Dropout(rate=0.3)(down2pool)


    down4 = Conv2D(start_neurons*8, k_size, activation='relu', padding='same', dilation_rate=dilate_rate,kernel_initializer='he_normal')(down3pool)
    b7 = BatchNormalization()(down4)
    #b5 = Dropout(rate=0.3)(b5)
    down4 = Conv2D(start_neurons*8, k_size, activation='relu', padding='same',dilation_rate=dilate_rate, kernel_initializer='he_normal')(b7)
    b8 = BatchNormalization()(down4)
    #b6 = Dropout(rate=0.3)(b6)
    down4pool = MaxPooling2D((2, 2), strides=(2, 2))(b8)
    #down3pool = Dropout(rate=0.3)(down3pool)

    if dilate == 1:
    # stacked dilated convolution at the bottleneck
        dilate1 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=1, kernel_initializer='he_normal')(down4pool)
        b9 = BatchNormalization()(dilate1)
        #b7 = Dropout(rate=0.3)(b7)
        dilate2 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(b9)
        b10 = BatchNormalization()(dilate2)
        #b8 = Dropout(rate=0.3)(b8)
        dilate3 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(b10)
        b11 = BatchNormalization()(dilate3)
        #b9 = Dropout(rate=0.3)(b9)
        dilate4 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b11)
        b12 = BatchNormalization()(dilate4)
        #b10 = Dropout(rate=0.3)(b10)
        #dilate5 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b12)
        #b13 = BatchNormalization()(dilate5)
        #b11 = Dropout(rate=0.3)(b11)
        #dilate6 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', dilation_rate=32, kernel_initializer='he_normal')(b13)
        if addition == 1:
            dilate_all_added = add([dilate1, dilate2, dilate3, dilate4])#, dilate5, dilate6])
            #up3 = upsample((2, 2))(dilate_all_added)
            up4 = upsample(start_neurons * 8, k_size, strides=(2, 2), padding="same")(dilate_all_added)
        else:
            #up3 = upsample((2, 2))(dilate6)
            up4 = upsample(start_neurons * 8, k_size, strides=(2, 2), padding="same")(dilate6)
    else:
        dilate1 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(down3pool)
        b9 = BatchNormalization()(dilate1)
        #b7 = Dropout(rate=0.3)(b7)
        dilate2 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b9)
        b10 = BatchNormalization()(dilate2)
        #b8 = Dropout(rate=0.3)(b8)
        dilate3 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b10)
        b11 = BatchNormalization()(dilate3)
        #b9 = Dropout(rate=0.3)(b9)
        dilate4 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b11)
        b12 = BatchNormalization()(dilate4)
        #b10 = Dropout(rate=0.3)(b10)
        dilate5 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b12)
        b13 = BatchNormalization()(dilate5)
        #b11 = Dropout(rate=0.3)(b11)
        dilate6 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b13)
        if addition ==1:
            dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
            #up3 = upsample((2, 2))(dilate_all_added)
            up4 = upsample(start_neurons * 8, k_size, strides=(2, 2), padding="same")(dilate_all_added)
        else:
            #up3 = upsample((2, 2))(dilate6)
            up4 = upsample(start_neurons * 8, k_size, strides=(2, 2), padding="same")(dilate6)
    up4 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    up4 = concatenate([down4, up4])
    b14 = BatchNormalization()(up4)
    #b12 = Dropout(rate=0.3)(b12)
    up4 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b14)
    b15 = BatchNormalization()(up4)
    up4 = Conv2D(start_neurons*8,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b15)

    #up2 = upsample((2, 2))(up3)
    up3 = upsample(start_neurons * 4, k_size, strides=(2, 2), padding="same")(up4)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
    up3 = concatenate([down3, up3])
    b16 = BatchNormalization()(up3)
    #b14 = Dropout(rate=0.3)(b14)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b16)
    b17 = BatchNormalization()(up3)
    up3 = Conv2D(start_neurons*4,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b17)

    #up2 = upsample((2, 2))(up3)
    up2 = upsample(start_neurons * 2, k_size, strides=(2, 2), padding="same")(up3)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    up2 = concatenate([down2, up2])
    b18 = BatchNormalization()(up2)
    #b14 = Dropout(rate=0.3)(b14)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b18)
    b19 = BatchNormalization()(up2)
    up2 = Conv2D(start_neurons*2,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b19)

    #up1 = upsample((2, 2))(up2)
    up1 = upsample(start_neurons, k_size, strides=(2, 2), padding="same")(up2)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    up1 = concatenate([down1, up1])
    b20 = BatchNormalization()(up1)
    #b16 = Dropout(rate=0.3)(b16)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b20)
    b21 = BatchNormalization()(up1)
    up1 = Conv2D(start_neurons,k_size, activation='relu', padding='same', kernel_initializer='he_normal')(b21)
    b22 = BatchNormalization()(up1)

    output_layer = Conv2D(1, 1, activation='relu')(b22)

    model = Model(input=inputs, output=output_layer)
    if dc==1:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    elif dc==0:
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[mean_iou])
    return model
