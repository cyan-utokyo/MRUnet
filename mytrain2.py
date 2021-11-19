from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,MaxPooling3D, UpSampling2D, Dropout, Bidirectional, BatchNormalization
from keras.layers import concatenate, Conv2DTranspose,Conv3DTranspose
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,Callback
from keras.utils import Sequence
from keras.models import model_from_json
import os
import os.path
import glob
import cv2
from make_512_dataset import *
from att_unet import *
from ori_unet import *
from skimage.io import imsave
import datetime
import json
import warnings
import argparse

warnings.simplefilter('ignore')



EARLY_STOP_PATIENCE = 20
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE=5
REDUCE_LR_MIN_LR = 1e-7
#############################################
# make parser
parser = argparse.ArgumentParser(
            prog='mytrain2.py',
            usage='Demonstration of argparser',
            description='description',
            epilog='end',
            add_help=True,
            )

# add arugument


parser.add_argument('-t', '--testname',type=str)
parser.add_argument('-e', '--epochs',type=int)
parser.add_argument('-n', '--start_neurons',type=int)
parser.add_argument('-k', '--kernel_size',type=int)
parser.add_argument('-b', '--batch_size',type=int)
parser.add_argument('-v', '--valid_case_no',type=int,default=0)
parser.add_argument('-m', '--mode',default="train")
parser.add_argument('-nn', '--network_no',default=0)
parser.add_argument('-lr', '--learning_rate',default=1e-5)
parser.add_argument('-d', '--debug',default=0)
parser.add_argument('-s', '--source',default=0)
parser.add_argument('-p', '--working_path')

args = parser.parse_args()

start_neurons = int(args.start_neurons)
k_size = (int(args.kernel_size),int(args.kernel_size))
epochs = int(args.epochs)
batch_size = int(args.batch_size)
testname = args.testname
mode = args.mode
valid_case_no = int(args.valid_case_no)
network_no = int(args.network_no)
LR = float(args.learning_rate)
debug = int(args.debug)
source = int(args.source) # 1:MRI, 0:CT
working_path = args.working_path
###############################################
print ("="*10," args ","="*10)
print ("testname:",testname)
print ("epochs:",epochs)
print ("start_neurons:",start_neurons)
print ("k_size:",k_size)
print ("batch_size:",batch_size)
print ("mode:",mode)
print ("network no.:",network_no)
print ("valid_case_no:",valid_case_no, " (for -1, use validation data in CT_valid/)")
print ("debug:",debug)
print ("source:",source)
print ("working_path:",working_path)
print ("="*10," args ","="*10)


#############################################


root_dir ="/src/MRA/in-house/"
test_dir_path ="/home/chen/src/code/MR_Unet/"

model_name = 'unet.hdf5'

#if source ==0:
#    valid_path = root_dir+"CT_valid/"
#    train_path = root_dir+"CT/"
#    test_path = root_dir+"CT_test/"
#    img_size = 512
#elif source ==1:
#    print ("MRI!!")
#    valid_path = root_dir+"MRI_valid/"
#    train_path = root_dir+"MRI/"
#    test_path = root_dir+"MRI_test/"
#    img_size = 384
train_path = root_dir
img_size = 384
smooth = 1.


#def vmodeler_mask(res,rootdir,dirname,start=0):
#    save_img = mkdir(rootdir, dirname)
#    end = start+len(res)
#    for i in range(start, end):
#        img3=np.ndarray((512,512,3))
#        img = res[i,:,:,0]
#        img = img>0.5
#        img = np.around(img)
#        img3[:,:,0]=img
#        img3[:,:,1]=np.zeros((512,512))
#        img3[:,:,2]=np.zeros((512,512))
#        imsave(save_img+"/z{:0=4}.bmp".format(i-start),img3)# CT:+401

def train(valid_case_no =-1, network_no=0):
    train_case_list = glob.glob(train_path+"*/")[:7]
    valid_case_list = glob.glob(train_path+"*/")[7:]

    print ("="*10," valid & train set ","="*10)
    print ("valid case:",valid_case_list)
    print ("train case:",train_case_list)
    print ("="*10," valid & train set ","="*10)

    raw_all_train,seg_all_train=get_img_list(train_case_list)
    raw_all_valid,seg_all_valid=get_img_list(valid_case_list)

    train_gen = gen_chunk(raw_all_train,seg_all_train,batch_size=batch_size)
    train_aug_gen = gen_aug_chunk(train_gen)
    valid_gen=gen_chunk(raw_all_valid,seg_all_valid)

    raw_len = len(raw_all_train)
    valid_raw_len = len(raw_all_valid)
    print ("lengths: ",raw_len,"(train) ",valid_raw_len,"(valid)")
    print ("network_no in train():", network_no,type(network_no))

    if (network_no==0):
        print ("network_no == 0, 4blocks")
        model = get_unet_4block(img_size=img_size, start_neurons = start_neurons, k_size=k_size, learning_rate = LR, dc=1,  UPSAMPLE_MODE='DECONV')

    #working_path = mkdir(test_dir_path,testname)
    #working_path = os.getcwd()


    json_model = model.to_json()
    with open(working_path + 'model.json', 'w') as f:
        json.dump(json_model, f, indent=4)

    logs = glob.glob(working_path + "log_*.log")
    logs_len = len(logs)
    log_name = "log_{}.log".format(logs_len)
    try:
        model.load_weights(working_path + model_name)
    except OSError:
        print ("No weights to be loaded.")

    #train_dir_list = glob.glob(train_path+"*/") # old method to extract cases according to the dir(CT/CT_valid)
    #valid_dir_list = glob.glob(valid_path+"*/")

    with open (working_path + "data_record.txt", "w") as f2:
        f2.write("Train:\n")
        for file_name in train_case_list:
            f2.write(file_name)
            f2.write("\n")
        f2.write("Valid:\n")
        for file_name in valid_case_list:
            f2.write(file_name)
            f2.write("\n")

    if os.uname()[1] =='chen-zz':
        plot_model(model, to_file=working_path + 'model.png', show_shapes=True)

    model_checkpoint = ModelCheckpoint(working_path+model_name, monitor='val_dice_coef',mode="max",verbose=1, save_best_only=True)
    #model_checkpoint = ModelCheckpoint(working_path+model_name, monitor='val_mean_iou',mode="max",verbose=1, save_best_only=True)

    ### NEW added by chen. ###
    print ("early stopping patience:",EARLY_STOP_PATIENCE)
    print ("reduce LR factor:",REDUCE_LR_FACTOR)
    print ("reduce LR patience:",REDUCE_LR_PATIENCE)
    early_stopping = EarlyStopping(monitor="val_dice_coef",mode="max",patience=EARLY_STOP_PATIENCE,verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_dice_coef",mode="max",factor=REDUCE_LR_FACTOR,patience=REDUCE_LR_PATIENCE,min_lr=REDUCE_LR_MIN_LR,verbose=1,epsilon=0.005)
    #early_stopping = EarlyStopping(monitor="val_mean_iou",mode="max",patience=EARLY_STOP_PATIENCE,verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor="val_mean_iou",mode="max",factor=REDUCE_LR_FACTOR,patience=REDUCE_LR_PATIENCE,min_lr=REDUCE_LR_MIN_LR,verbose=1,epsilon=0.005)

    csv_logger = CSVLogger(working_path+log_name)

    #history = model.fit(imgs_train, imgs_mask_train, validation_data=[imgs_vali,imgs_mask_vali], batch_size=2, epochs=500, verbose=1, shuffle=True,callbacks=[early_stopping,model_checkpoint,reduce_lr,csv_logger])

    model.summary()

    history = model.fit_generator(
            train_aug_gen,
            #train_gen,
            steps_per_epoch = raw_len/batch_size,
            epochs=epochs,
            verbose=1,
            validation_data= valid_gen,
            validation_steps = valid_raw_len/batch_size,
            callbacks=[early_stopping,model_checkpoint,reduce_lr,csv_logger]
            )
    print ("training for ",epochs," epochs.")

def predict(load_model_from="json"):
    #working_path = mkdir(test_dir_path,testname)
    #working_path = os.getcwd()
    if load_model_from == "json":
        json_file = open(working_path + 'model.json', 'r')
        json_load=json.load(json_file)
        model = model_from_json(json_load)
        model.compile(optimizer=Adam(lr=5e-5), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        model = get_unet_4block()
    model.load_weights(working_path + model_name)
    test_case_list = glob.glob(test_path+"*/")
    raw_all_test,seg_all_test=get_img_list(test_case_list)
    with open(working_path + "test_list.txt", "w") as f:
        for img_name in raw_all_test:
            f.write(img_name)
            f.write("\n")
    test_raw_len = len(raw_all_test)
    test_gen=gen_forward_chunk(raw_all_test,seg_all_test)
    score = model.evaluate_generator(test_gen,steps=test_raw_len)
    print ("SCORE:", score)
    #predict_mask = model.predict_generator(test_gen,steps=len(test_gen),verbose=1)
    #np.save(working_path+"result.npy",predict_mask)
    #print ("saved. npy shape is:",predict_mask.shape,"while loaded ",len(test_gen),"imgs.")


if __name__ == '__main__':

    if mode =="train":
        train(valid_case_no =valid_case_no, network_no=network_no)
        # predict()

    elif mode=="predict":
        predict()
