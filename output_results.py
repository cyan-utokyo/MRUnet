import numpy as np
import matplotlib.pyplot as plt
from make_dataset import *
# import seaborn as sns
from skimage.io import imsave

def output_imgs(raws, tars, ress):
    #dir = mkdir(path, dir_name)
    dir = "/home/chen/u-net/ct2/results/"
    for i in range(raws.shape[0]):
        raw = raws[i,:,:,0]
        tar = tars[i,:,:,0]
        res = ress[i,:,:,0]
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.title.set_text('Raw')
        ax1 = plt.imshow(raw)
        ax2 = fig.add_subplot(1,3,2)
        ax2.title.set_text('Target')
        ax2 = plt.imshow(tar)
        ax3 = fig.add_subplot(1,3,3)
        ax3.title.set_text('Results')
        ax3 = plt.imshow(res)
        #plt.show()
        plt.savefig(dir + "res_{:0=4}.png".format(i))


def vmodeler_mask(res,rootdir,dirname,start=0):
    save_img = mkdir(rootdir, dirname)
    end = start+len(res)
    for i in range(start, end):
        img3=np.ndarray((512,512,3))
        img = res[i,:,:,0]
        img = img>0.5
        img = np.around(img)
        img3[:,:,0]=img
        img3[:,:,1]=np.zeros((512,512))
        img3[:,:,2]=np.zeros((512,512))
        imsave(save_img+"/z{:0=4}.bmp".format(i-start),img3)# CT:+401

res = np.load("/home/chen/u-net/ct6/res_mask.npy")
print (res.shape)


output_imgs(img,tar,res)

#ress = np.load("/Users/chenyan/ct4/res_mask_hk.npy")
print (res.shape)
"""
if len(res.shape) == 5:
    ress = res.reshape([res.shape[0]*res.shape[1],res.shape[2],res.shape[3],res.shape[4]])
vmodeler_mask(res, rootdir="/Users/chenyan/ct2/",dirname="res4")
"""



#tars = np.load("/home/chen/u-net/ct2/tar_mask.npy")
#ress = np.load("/home/chen/u-net/ct2/res_mask.npy")
#output_imgs(raws, tars, ress)
