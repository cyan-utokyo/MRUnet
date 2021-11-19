from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,MaxPooling3D, UpSampling2D, Dropout, Bidirectional, BatchNormalization
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

def get_lstm_unet():
    start_neurons = 32 # in the original case 32

    inputs = Input((slices, img_size, img_size, 1))

    # convlstm = ConvLSTM2D(int(start_neurons/2), (3, 3), activation="relu", padding="same", return_sequences = True)(inputs)

    conv1 = ConvLSTM2D(start_neurons * 1, (3, 3), activation="relu", padding="same", return_sequences = True)(inputs)
    conv1 = ConvLSTM2D(start_neurons * 1, (3, 3), activation="relu", padding="same", return_sequences = True)(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    pool1 = TimeDistributed(Dropout(0.25))(pool1)

    conv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(pool1)
    conv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(conv2)
    #conv2 = ConvLSTM2D(start_neurons * 2, (3, 3), activation="relu", padding="same", return_sequences = True)(pool1)
    #conv2 = ConvLSTM2D(start_neurons * 2, (3, 3), activation="relu", padding="same", return_sequences = True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(pool2)
    conv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(conv3)
    #conv3 = ConvLSTM2D(start_neurons * 4, (3, 3), activation="relu", padding="same", return_sequences = True)(pool2)
    #conv3 = ConvLSTM2D(start_neurons * 4, (3, 3), activation="relu", padding="same", return_sequences = True)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(pool3)
    conv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(conv4)
    #conv4 = ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True)(pool3)
    #conv4 = ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True)(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = TimeDistributed(Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same"))(pool4)
    convm = TimeDistributed(Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same"))(convm)
    # 8 -> 16
    deconv4 = TimeDistributed(Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same"))(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(uconv4)
    uconv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(uconv4)

    # 16 -> 32
    deconv3 = TimeDistributed(Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same"))(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(uconv3)
    uconv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(uconv3)

    # 32 -> 64
    deconv2 = TimeDistributed(Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same"))(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(uconv2)
    uconv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(uconv2)

    # 64 -> 128
    deconv1 = TimeDistributed(Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same"))(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(uconv1)
    uconv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(uconv1)

    uconv1 = Dropout(0.5)(uconv1)
    output_layer = TimeDistributed(Conv2D(1, (1,1), padding="same", activation="sigmoid"))(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    #output_layer = TimeDistributed(Conv2D(1, (1,1), padding="same", activation="softmax"))(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=2e-4), loss="binary_crossentropy", metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

    return model

def get_lstm_unet_2():
    start_neurons = 32 # in the original case 32

    inputs = Input((slices, img_size, img_size, 1))

    #conv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(inputs)
    #conv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(conv1)
    conv1 = ConvLSTM2D(start_neurons * 1, (3, 3), activation="relu", padding="same", return_sequences = True)(inputs)
    conv1 = ConvLSTM2D(start_neurons * 1, (3, 3), activation="relu", padding="same", return_sequences = True)(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    pool1 = TimeDistributed(Dropout(0.25))(pool1)

    #conv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(pool1)
    #conv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(conv2)
    conv2 = ConvLSTM2D(start_neurons * 2, (3, 3), activation="relu", padding="same", return_sequences = True)(pool1)
    conv2 = ConvLSTM2D(start_neurons * 2, (3, 3), activation="relu", padding="same", return_sequences = True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    pool2 = TimeDistributed(Dropout(0.5))(pool2)

    #conv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(pool2)
    #conv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(conv3)
    conv3 = ConvLSTM2D(start_neurons * 4, (3, 3), activation="relu", padding="same", return_sequences = True)(pool2)
    conv3 = ConvLSTM2D(start_neurons * 4, (3, 3), activation="relu", padding="same", return_sequences = True)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    pool3 = TimeDistributed(Dropout(0.5))(pool3)

    #conv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(pool3)
    #conv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(conv4)
    conv4 = ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True)(pool3)
    conv4 = ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True)(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)
    pool4 = TimeDistributed(Dropout(0.5))(pool4)

    # Middle
    convm = TimeDistributed(Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same"))(pool4)
    convm = TimeDistributed(Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same"))(convm)
    #convm = Bidirectional(ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True))(pool4)
    #convm = Bidirectional(ConvLSTM2D(start_neurons * 8, (3, 3), activation="relu", padding="same", return_sequences = True))(convm)

    # 8 -> 16
    deconv4 = TimeDistributed(Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same"))(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = TimeDistributed(Dropout(0.5))(uconv4)
    uconv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(uconv4)
    uconv4 = TimeDistributed(Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same"))(uconv4)

    # 16 -> 32
    deconv3 = TimeDistributed(Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same"))(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = TimeDistributed(Dropout(0.5))(uconv3)
    uconv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(uconv3)
    uconv3 = TimeDistributed(Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same"))(uconv3)

    # 32 -> 64
    deconv2 = TimeDistributed(Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same"))(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = TimeDistributed(Dropout(0.5))(uconv2)
    uconv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(uconv2)
    uconv2 = TimeDistributed(Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same"))(uconv2)

    # 64 -> 128
    deconv1 = TimeDistributed(Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same"))(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = TimeDistributed(Dropout(0.5))(uconv1)
    uconv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(uconv1)
    uconv1 = TimeDistributed(Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same"))(uconv1)

    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = TimeDistributed(Conv2D(1, (1,1), padding="same", activation="sigmoid"))(uconv1)
    #output_layer = TimeDistributed(Conv2D(1, (1,1), padding="same", activation="softmax"))(uconv1)

    # model = Model(input=inputs, output=conv10)
    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=2e-4), loss="binary_crossentropy", metrics=[dice_coef])
    # model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

    return model
