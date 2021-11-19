import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from keras.models import model_from_json
from keras.optimizers import Adam
import json
from unet.losses import *
warnings.filterwarnings('ignore')

img_size = 512
start_neurons = 32
k_size =3
LR = 1e-3
working_path = os.getcwd()+"/"
model_name = 'unet.hdf5'
print (working_path)

json_file = open(working_path + 'model.json', 'r')
json_load=json.load(json_file)
model = model_from_json(json_load)
model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
model.load_weights(working_path + model_name)
model.summary()

# summarize filter shapes
conv_l = []
for layer in model.layers:
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)

from unet.make_512_dataset import gen_chunk, gen_aug_chunk, get_img_list
import glob

valid_case_no=-1
debug=1
batch_size =4

root_dir = "/home/chen/material/"
test_dir_path ="/home/chen/u-net/"

test_path = root_dir+"CT_valid/"

test_case_list = glob.glob(test_path+"*/")

print ("="*10," test set ","="*10)
print ("test case:",test_case_list)
print ("="*10," test set ","="*10)


with open (working_path + "test_data_record.txt", "w") as f2:
    f2.write("Test:\n")
    for file_name in test_case_list:
        f2.write(file_name)
        f2.write("\n")

raw_all_test,seg_all_test=get_img_list(test_case_list)
test_gen=gen_chunk(raw_all_test,seg_all_test)
test_raw_len = len(raw_all_test)
print ("lengths: ",test_raw_len,"(test)")

score = model.evaluate_generator(test_gen,steps=test_raw_len)
print ("SCORE:", score)
