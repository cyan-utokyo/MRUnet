from __future__ import print_function
import numpy as np
import os
import glob
import cv2
from make_512_dataset import *
from att_unet import *
from ori_unet import *
from skimage.io import imsave
from skimage.color import gray2rgb
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.color import rgb2gray, rgb2hsv, label2rgb
import datetime
import json
import warnings
import argparse
from ori_unet import *
import json

img_size=512
start_neurons=32
k_size=3
LR=1e-5

transfer_name = "transfer_2020-06-14-07-36/"
comment = "CT_valid, interval=100,"

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + 1.)


#########

TF=1

if TF == 1:
    model_name = 'transfer.hdf5'
    transfer_dir = os.getcwd()[:-4]+transfer_name
    print ("make TRANSFER-ed images")
    man_list_tar = []
    man_list_res =[]
    with open(transfer_dir + "train_transfer_list.txt") as f:
        man_list = f.readlines()
    for i in range(len(man_list)):
        man_list_tar.append(man_list[i][:-1])
        man_list[i] = [man_list[i][:-1].split('/')[5],man_list[i][:-1].split('/')[7]]
        man_list_res.append(transfer_dir+"{}/{}".format(man_list[i][0],man_list[i][1]))
elif TF == 0:
    model_name = 'unet.hdf5'
    transfer_dir = os.getcwd()[:-4]
    print ("make non-TRANSFER-ed images")

print (transfer_dir)

#########

if (os.uname()[1])=='chen-zz':
    root_dir = "/home/chen/material/"
    test_dir_path ="/home/chen/u-net/"
else:
    #print (os.uname()[1]," is hostname.")
    root_dir ="/workspace/unet/materials/"
    test_dir_path ="/workspace/unet/"
test_path = root_dir+"CT_valid/"
test_case_list = glob.glob(test_path+"*/")
print (test_case_list)
raw_all_test,seg_all_test=get_img_list(test_case_list[:3])

#########

if glob.glob(transfer_dir+'*/') == []:
    # forward.
    model = get_unet_4block(img_size=img_size, start_neurons = start_neurons, k_size=k_size, learning_rate = LR)
    model.load_weights(transfer_dir+model_name)

    results = []

    for i in range(len(raw_all_test)):
        filename = raw_all_test[i]

        case_name=filename.split('/')[5]
        img_name=filename.split('/')[-1]
        save_path = mkdir(transfer_dir, case_name)

        in_img = read_a_ct(raw_all_test[i],"raw")/255.
        in_mask = read_a_ct(seg_all_test[i],"segmentation")
        in_mask=in_mask[np.newaxis,:,:,np.newaxis]
        local_results = []
        local_scores = []
        for j in range(5):
            gamma = np.random.rand()*2.5
            disk_num = np.random.randint(15,25)
            selem = disk(disk_num)
            in_img_gamma = exposure.adjust_gamma(in_img, gamma)
            in_img_gamma_eq = rank.equalize(in_img_gamma, selem=selem)
            in_img_gamma_eq=in_img_gamma_eq[np.newaxis,:,:,np.newaxis]
            result = model.predict(in_img_gamma_eq)
            score = model.evaluate(in_img_gamma_eq, in_mask)
            local_results.append(result)
            local_scores.append(score)
            #print (i,gamma,disk_num,score)
        max_score_idx = local_scores.index(max(local_scores))
        result = local_results[max_score_idx]

        DCs=[]
        results_threshold=[]
        for j in np.linspace(0,0.9,10):
            result_threshold = result>j
            DCs.append(dice_coef_np(result_threshold, in_mask))
            results_threshold.append(result_threshold)
        max_DC = DCs.index(max(DCs))
        result = results_threshold[max_DC]

        red_img = np.zeros((512,512,3))
        red_img[:,:,2] = result[0,:,:,0]*255
        cv2.imwrite(save_path+img_name, red_img)
        results.append(result)
#########

from tqdm import tqdm

cal_dc = False

valid_case_paths = ['CEA4_postope_CTA/', 'cas2_post/', 'Yamagata_Lt.ICA_stenosis/', 'CEA2_CT/']
test_case_paths = ['mcs1_post/', 'CEA9_preope_CTA/' , 'CAS2_pre/']

case_path =valid_case_paths[0]
imgs_res_path = glob.glob(transfer_dir+case_path+"*.bmp")
imgs_res_path = sorted(imgs_res_path)

imgs_tar_path = glob.glob(test_path+case_path+"segmentation/"+"*.bmp")
imgs_tar_path = sorted(imgs_tar_path)
imgs_res = []
imgs_tar =[]
for img_res_path,img_tar_path in tqdm(zip(imgs_res_path,imgs_tar_path)):
    if (img_res_path in man_list_res) and cal_dc ==True:
        print (img_res_path)
        continue
    img_res = read_a_ct(img_res_path,"segmentation")
    img_tar = read_a_ct(img_tar_path,"segmentation")
    imgs_tar.append(img_tar)
    imgs_res.append(img_res)
imgs_res_ori = np.array(imgs_res)
imgs_tar = np.array(imgs_tar)

print (case_path,len(imgs_res_path))

print (imgs_res_ori.shape)

#########
from copy import copy
from skimage import morphology

imgs_res = copy(imgs_res_ori)
print (imgs_res.shape)
img_mor = []

# 双方向online更新扩张/收缩, 或3D扩张收缩
d0 = 7
d1 = 7
d2 = 3
for i in tqdm(range(imgs_res.shape[1])):
    imgs_res[:,i,:] = morphology.binary_dilation(imgs_res[:,i,:], morphology.diamond(d0)).astype(np.uint8)
    imgs_res[:,i,:] = morphology.binary_erosion(imgs_res[:,i,:], morphology.diamond(d0)).astype(np.uint8)
    imgs_res[:,:,i] = morphology.binary_dilation(imgs_res[:,:,i], morphology.diamond(d1)).astype(np.uint8)
    imgs_res[:,:,i] = morphology.binary_erosion(imgs_res[:,:,i], morphology.diamond(d1)).astype(np.uint8)
    # denoise
    imgs_res[:,i,:] = morphology.binary_erosion(imgs_res[:,i,:], morphology.diamond(d2)).astype(np.uint8)
    imgs_res[:,i,:] = morphology.binary_dilation(imgs_res[:,i,:], morphology.diamond(d2)).astype(np.uint8)


#########

# marching cubes + denoise + hausdorff comparison.
# skip if only calculate DC.
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


imgs_res_ori_2 = imgs_res_ori.transpose(2,1,0)
imgs_res_2 = imgs_res.transpose(2,1,0)
imgs_tar_2 = imgs_tar.transpose(2,1,0)

verts_tar, faces_tar, normals_tar, values_tar = measure.marching_cubes_lewiner(imgs_tar_2, 0)
mesh_tar = Poly3DCollection(verts_tar[faces_tar])
mesh_tar.set_edgecolor('lightgreen')
mesh_tar.set_facecolor('darkgreen')

verts_mor, faces_mor, normals_mor, values_mor = measure.marching_cubes_lewiner(imgs_res_2, 0)
mesh_mor = Poly3DCollection(verts_mor[faces_mor])

verts_ori, faces_ori, normals_ori, values_ori = measure.marching_cubes_lewiner(imgs_res_ori_2, 0)
mesh_ori = Poly3DCollection(verts_ori[faces_ori])


fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(131, projection='3d')
#mesh_tar.set_edgecolor('k')
ax.add_collection3d(mesh_tar)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_xlim(75, 400)
ax.set_ylim(50, 300)
ax.set_zlim(0, 500)
ax.view_init(elev=10, azim=60)

ax2 = fig.add_subplot(132, projection='3d')
#mesh_mor.set_edgecolor('k')
ax2.add_collection3d(mesh_mor)
ax2.set_xlabel("x-axis")
ax2.set_ylabel("y-axis")
ax2.set_zlabel("z-axis")
ax2.set_xlim(75, 400)
ax2.set_ylim(50, 300)
ax2.set_zlim(0, 500)
ax2.view_init(elev=10, azim=60)

ax3 = fig.add_subplot(133, projection='3d')
#mesh_mor.set_edgecolor('k')
ax3.add_collection3d(mesh_ori)
ax3.set_xlabel("x-axis")
ax3.set_ylabel("y-axis")
ax3.set_zlabel("z-axis")
ax3.set_xlim(75, 400)
ax3.set_ylim(50, 300)
ax3.set_zlim(0, 500)
ax3.view_init(elev=10, azim=60)

plt.tight_layout()
# plt.show()
plt.savefig(transfer_dir+case_path+"vessel.png")


print (transfer_name)
print (comment)
print (dice_coef_np(imgs_tar,imgs_res))
print (dice_coef_np(imgs_tar,imgs_res_ori))

#########
#########
