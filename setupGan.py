# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:12:16 2021

@author: Alkios
"""



import os
import PIL
import time
import glob
import struct
import imageio
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from tensorflow.keras import layers


# =============================================================================
# 
# =============================================================================


def read_double_binary(fichier): #read_double_binary(file)
    with open(fichier, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    
    data = struct.unpack( "d" * ((len(fileContent)) // 8), fileContent )
    
    return data


def write_double_binary(fichier, data): #write_double_binary(file, data)
    try:
        with open(fichier, 'wb') as file:
            for elem in data:
                file.write(struct.pack('<d', elem))
    except IOError:
        print('Erreur d\'Ã©criture.')


# =============================================================================
# 
# =============================================================================


# ST = ['05', '1', '5']
ST = ['1']


train_images = np.array([], dtype=np.int32).reshape(0, 128, 128)

for St in ST:
    print(St)
    train_images = np.concatenate((train_images, np.reshape(read_double_binary('C://Users/Alkios/Downloads/ProjetM1/DataGan/Cube_N512_st'+str(St)+'_'+str(256-128)), (7680, 128, 128))))

    

new_train_images = []

for i in train_images :
    
    i = i.reshape(128,128)
    
    hflipped = np.fliplr(i)
    vflipped = np.flipud(i)
    hvflipped = np.flipud(hflipped)
    
    new_train_images.append(hflipped)
    new_train_images.append(vflipped)
    new_train_images.append(hvflipped)


new_train_images = np.array(new_train_images,float)

train_images = np.concatenate((train_images,new_train_images))


    
#MEAN = np.mean(train_images)
#STD = np.std(train_images)

train_images = train_images.reshape(train_images.shape[0], 128, 128, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 



plt.imshow(train_images[30000].reshape(128,128), cmap = 'hot')


from mpl_toolkits.axes_grid1 import make_axes_locatable


ax = plt.subplot()
im = ax.imshow(train_images[30000].reshape(128,128)*127.5+127.5, cmap = 'hot')

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

plt.show()
