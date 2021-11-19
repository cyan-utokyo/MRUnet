import numpy as np
import glob
import seaborn as sns
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from numpy.random import *
#from make_dataset import *
import datetime
from skimage.color import gray2rgb
import os
from skimage.util import montage
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.color import rgb2gray, rgb2hsv, label2rgb

if (os.uname()[1])=='chen-zz':
    from keras.preprocessing.image import ImageDataGenerator
else:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

BI = 0

d_gen = ImageDataGenerator(rotation_range=20,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           shear_range=0.1,
                           zoom_range=0.3,
                           fill_mode='constant',
                           cval = 0 ,
                           horizontal_flip=True,
                           vertical_flip=False)

#def mkdir(super_path,testname):
    ###dir_path = test_dir_path+"{}/".format(testname)
    #dir_path = super_path+"{}/".format(testname)
    #if os.path.exists(dir_path)==False:
        #print("making new directory {}...".format(dir_path))
        #os.mkdir(dir_path) # pythonからフォルダを作らないように
    #else:
        #print("generating in directory {}...".format(dir_path))
    #return dir_path


def read_a_ct(img_path,raw_or_seg="raw",bi=0):
    #img = io.imread(img_path)
    
    img = cv2.imread(img_path)
    if raw_or_seg =="raw":
        #if bi == 1:
            #img = cv2.bilateralFilter(img, 15, 20, 30)
            #img = cv2.bilateralFilter(img, 15, 20, 30)
        #img = cv2.bilateralFilter(img, 15, 20, 20)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #print "becase it is a raw, turn img.shape into: ",img.shape
    elif raw_or_seg == "segmentation":
        img = cut_mask(img)
    return img


def cut_mask(seg):
    seg = rgb2hsv(seg)
    mask = seg[:,:,1]
    return mask

def get_img_list(case_list):# original:(rootpath)
    #case_list = glob.glob(root_path+"*/")
    raw_dir_list = []
    seg_dir_list = []
    for m in range(len(case_list)):
        raw_dir_name = "raw/"
        seg_dir_name = "masked/"
        raw_dir_list.append(raw_dir_name)
        seg_dir_list.append(seg_dir_name)
    seg_all = []
    raw_all = []
    for i in range(len(case_list)):
        seg_all += glob.glob(case_list[i]+seg_dir_list[i]+"*.bmp")
    seg_all = sorted(seg_all)
    for j in range(len(seg_all)):
        raw_path = seg_all[j].replace('segmentation','raw')
        # raw_path = raw_path.replace('norm_seg','norm_raw')
        raw_all.append(raw_path)
    return raw_all, seg_all # path, eg.:home/chen/material/CT/cas2_post/raw(segmentation)/z0500.bmp

def gen_chunk(raw_all, seg_all, batch_size = 4):
    while True:
        img_batch = []
        mask_batch = []
        gamma = np.random.rand()*2.5
        disk_num = np.random.randint(15,25)
        selem = disk(disk_num)
        for _ in range(batch_size):
            i = np.random.randint(0,len(raw_all))
            in_img = read_a_ct(raw_all[i],"raw", bi=BI)/255.
            in_img_gamma = exposure.adjust_gamma(in_img, gamma)
            if in_img.shape[0] != 512:
                in_img_gamma = cv2.resize(in_img_gamma,(384,384))
                img_batch += [in_img_gamma]
            else:
                in_img_gamma_eq = rank.equalize(in_img_gamma, selem=selem)
                img_batch += [in_img_gamma_eq]
            #in_img_gamma_eq = in_img_gamma
            in_mask = read_a_ct(seg_all[i],"segmentation")
            if in_mask.shape[0] != 512:
                in_mask = cv2.resize(in_mask,(384,384))
            mask_batch += [in_mask]
        yield np.stack(img_batch, 0)[:,:,:,np.newaxis], np.stack(mask_batch, 0)[:,:,:,np.newaxis]

def gen_forward_chunk(raw_all, seg_all):
    while True:
        img_batch = []
        mask_batch = []
        gamma = np.random.rand()*2.5
        disk_num = np.random.randint(15,25)
        selem = disk(disk_num)
        for i in range(len(raw_all)):
            in_img = read_a_ct(raw_all[i],"raw")/255.
            in_img_gamma = exposure.adjust_gamma(in_img, gamma)
            #in_img_gamma_eq = rank.equalize(in_img_gamma, selem=selem)
            in_mask = read_a_ct(seg_all[i],"segmentation")
            #img_batch += [in_img_gamma_eq]
            img_batch += [in_img_gamma]
            mask_batch += [in_mask]
        yield np.stack(img_batch, 0)[:,:,:,np.newaxis], np.stack(mask_batch, 0)[:,:,:,np.newaxis]


def gen_aug_chunk(in_gen):
    for i, (x_img, y_img) in enumerate(in_gen):
        SD = np.random.randint(0,1000)
        raw_gen = d_gen.flow(x_img, shuffle=True, seed=SD, batch_size = x_img.shape[0])
        msk_gen = d_gen.flow(y_img, shuffle=True, seed=SD, batch_size = x_img.shape[0])
        x_aug = next(raw_gen)
        y_aug = next(msk_gen)
        yield x_aug, y_aug


##############################################
# legacy functions that load data from .npy. #
# may use when training 3D or patch?         #
##############################################

def montage_nd(in_img):
    if len(in_img.shape)>3:
        return montage(np.stack([montage_nd(x_slice) for x_slice in in_img],0))
    elif len(in_img.shape)==3:
        return montage(in_img)
    else:
        warn('Input less than 3d image, returning original', RuntimeWarning)
        return in_img

def gen_chunk_from_npy(raws, masks, slice_count = 1, batch_size = 3):
    while True:
        i = np.random.choice(range(len(raws)))# ランダムにボリューム画像を選ぶ
        in_img = raws[i]
        in_mask = masks[i]
        img_batch = []
        mask_batch = []
        for _ in range(batch_size):
            s_idx = np.random.choice(range(in_img.shape[0]-slice_count))
            #img_batch += [in_img[s_idx:(s_idx+slice_count)]]
            #mask_batch += [in_mask[s_idx:(s_idx+slice_count)]]
            img_batch += [in_img[s_idx]]
            mask_batch += [in_mask[s_idx]]
        yield np.stack(img_batch, 0), np.stack(mask_batch, 0)

def gen_chunk_for_predict(raws,masks,raw_id, batch_size = 3):
    in_img = raws[raw_id]
    in_mask = masks[raw_id]
    s_idx = 0
    print ("s_idx:", s_idx)
    while True:
        #i = np.random.choice(range(len(raws)))
        img_batch = []
        mask_batch = []
        for _ in range(batch_size):
            #img_batch += [in_img[s_idx:(s_idx+slice_count)]]
            #mask_batch += [in_mask[s_idx:(s_idx+slice_count)]]
            img_batch += [in_img[s_idx]]
            mask_batch += [in_mask[s_idx]]
            if s_idx + 2*slice_count < in_img.shape[0]:
                s_idx = s_idx + slice_count
        yield np.stack(img_batch, 0), np.stack(mask_batch, 0)

def gen_aug_chunk_npy(in_gen):
    for i, (x_img, y_img) in enumerate(in_gen):
        xy_block = np.concatenate([x_img, y_img], 1).swapaxes(1, 4)[:, 0]
        #xy_block = np.concatenate([x_img, y_img], 1).swapaxes(1, 3)[:, 0]
        img_gen = d_gen.flow(xy_block, shuffle=True, seed=i, batch_size = x_img.shape[0])
        xy_scat = next(img_gen)
        # unblock
        xy_scat = np.expand_dims(xy_scat,1).swapaxes(1, 4)
        #xy_scat = np.expand_dims(xy_scat,1).swapaxes(1, 3)
        yield xy_scat[:, :xy_scat.shape[1]//2], xy_scat[:, xy_scat.shape[1]//2:]


def load_data_list(raw_path, mask_path):
    print ("="*30)
    print ("Mode: training")
    print ("loading raw in:", raw_path)
    print ("loading mask in:", mask_path)
    print ("="*30)
    raw_list = glob.glob(raw_path + "*/imgs_raw_gm.npy")
    mask_list = glob.glob(mask_path + "*/imgs_mask_gm.npy")

    raw = []
    mask = []
    print ("{} directories in raw_list".format(len(raw_list)))
    for i in range(len(raw_list)):
    #for i in range(2):
        print (raw_list[i])
        rawnpy = np.load(raw_list[i]).astype(np.float32)
        masknpy = np.load(mask_list[i]).astype(np.bool)
        raw_o =rawnpy[:,:,:,np.newaxis]
        mask_o = masknpy[:,:,:,np.newaxis]

        raw.append(raw_o)
        mask.append(mask_o)

    print ("raw length:",len(raw))
    return raw, mask

def load_training_data(raw_path, mask_path, sets=-1):
    # with small datasetf
    #slices = 5
    print ("="*30)
    print ("Mode: training")
    print ("loading raw in:", raw_path)
    print ("loading mask in:", mask_path)
    print ("="*30)

    raw_list = glob.glob(raw_path + "*/imgs_raw_gm.npy")
    mask_list = glob.glob(mask_path + "*/imgs_mask_gm.npy")

    raw = []
    mask = []
    #ske = []
    print ("{} directories in raw_list".format(len(raw_list)))
    #for i in range(1):
    if sets == -1:
        len_range = range(len(raw_list))
    elif sets in range(len(raw_list)):
        len_range = range(sets, sets+1)
    else:
        print ("out of range.")
        return
    for i in len_range:
        print ("load case:", raw_list[i])
        rawnpy = np.load(raw_list[i])
        masknpy = np.load(mask_list[i])
        print ("rawnpy shape:",rawnpy.shape)
        print ("masknpy shape:",masknpy.shape)
        raw.extend(rawnpy)
        mask.extend(masknpy)
    raw = np.array(raw)
    mask = np.array(mask)
    return raw[:,:,:,np.newaxis].astype(np.float32), mask[:,:,:,np.newaxis].astype(np.bool)

"""
print ("="*55)
print ("*"*10,"START:",datetime.datetime.now(),"*"*10)
dt1 = datetime.datetime.now()
raws, masks = load_data_list(training_raw_path,training_mask_path)
train_gen = gen_chunk(raws, masks)
train_aug_gen = gen_aug_chunk(train_gen)
x_out, y_out = next(train_aug_gen)

print(x_out.shape, y_out.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_nd(x_out[...,0]), cmap = 'bone')
ax1.set_title('In Batch')
ax2.imshow(montage_nd(y_out[...,0])cmap = 'Reds')
ax2.set_title('Out Batch')
plt.savefig("in_out_batch.png")
#png_list = glob.glob(test_mask_path+"sample_raws"+"/*.png")
#if png_list != 0:
#    for png in png_list:
#        os.remove(png)
#    print (len(png_list)," .png files have been deleted.")
#sample_path = mkdir(test_mask_path,"sample_raws")

dt2 = datetime.datetime.now()
print ("complete.(in {})".format(dt2-dt1))

print ("*"*11,"END:",datetime.datetime.now(),"*"*11)
print ("="*55)"""
