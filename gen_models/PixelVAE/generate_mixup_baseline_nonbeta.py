import os, sys
sys.path.append(os.getcwd())

import random

import numpy as np
import tensorflow as tf
import imageio
from imageio import imsave

import keras

import time
import functools

import sklearn
from sklearn.model_selection import train_test_split

DATASET = 'cifar10' # mnist_256
SETTINGS = '32px_cifar' # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big

OUT_DIR = DATASET + '_mixup_baseline_nonbeta'

if not os.path.isdir(OUT_DIR):
   os.makedirs(OUT_DIR)
   print "Created directory {}".format(OUT_DIR)

from keras.datasets import cifar10
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32
NUM_CLASSES = 10
(x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
   
x_train_set = x_train_set.transpose(0,3,1,2)
x_test_set = x_test_set.transpose(0,3,1,2)
    
seed = 333
x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)

from keras.utils import np_utils           
x_augmentation_set = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH)) #LEILEDIT: to enable .npy image saving
y_augmentation_set = np.zeros((1, 1, NUM_CLASSES)) #LEILEDIT: to enable .npy image saving.

#Function to translate numeric images into plots
def color_grid_vis(X, nh, nw, save_path):
  # from github.com/Newmu
  X = X.transpose(0,2,3,1)
  h, w = X[0].shape[:2]
  img = np.zeros((h*nh, w*nw, 3))
  for n, x in enumerate(X):
    j = n/nw
    i = n%nw
    img[j*h:j*h+h, i*w:i*w+w, :] = x
    imsave(OUT_DIR + '/' + save_path, img)            
                
numsamples = 11250
pvals = np.linspace(0.2, 0.8, num=4)
            
x_train_set_array = np.array(x_train_set)
y_train_set_array = np.array(y_train_set)  

for imagenum in range(numsamples):
    for class1 in range(NUM_CLASSES-1):
        idx1 = np.asarray(np.where(np.equal(class1, y_train_set))[0])
        x_trainsubset1 = x_train_set_array[idx1,:]
        y_trainsubset1 = y_train_set_array[idx1,:]
        x_trainsubset1 = x_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
        y_trainsubset1 = y_trainsubset1.reshape(-1, 1)
        
        for class2 in range(class1+1, NUM_CLASSES):
            idx2 = np.asarray(np.where(np.equal(class2, y_train_set))[0])
            x_trainsubset2 = x_train_set_array[idx2,:]
            y_trainsubset2 = y_train_set_array[idx2,:]
            x_trainsubset2 = x_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
            y_trainsubset2 = y_trainsubset2.reshape(-1, 1)
                    
            imageindex1 = random.sample(range(x_trainsubset1.shape[0]),1)
            imageindex2 = random.sample(range(x_trainsubset2.shape[0]),1)
                    
            # Draw the corresponding images and labels from the training data
            image1 = x_trainsubset1[imageindex1,:]
            image2 = x_trainsubset2[imageindex2,:]  
            label1 = y_trainsubset1[imageindex1,:]
            label2 = y_trainsubset2[imageindex2,:]
        
            #pvals = np.random.beta(0.2, 0.2, 1)
    
            # Reshape
            xarray1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
            xarray2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
            label1 = label1.reshape(1, 1)
            label2 = label2.reshape(1, 1)
                          
            # Change labels to matrix form before performing interpolations
            y1 = np_utils.to_categorical(label1, NUM_CLASSES) 
            y2 = np_utils.to_categorical(label2, NUM_CLASSES) 
                     
            # Combine the arrays and labels
            for p in pvals:
                new_xarray = np.multiply(p,xarray1) + np.multiply((1-p),xarray2)
                new_label = np.multiply(p,y1) + np.multiply((1-p),y2)
                new_label = new_label.reshape(1,1,NUM_CLASSES)

                x_augmentation_set = np.concatenate((x_augmentation_set, new_xarray), axis=0)#LEILAEDIT for .npy saving
                y_augmentation_set = np.concatenate((y_augmentation_set, new_label), axis=0)#LEILAEDIT for .npy saving
                              
x_augmentation_array = np.delete(x_augmentation_set, (0), axis=0)
y_augmentation_array = np.delete(y_augmentation_set, (0), axis=0)
            
x_augmentation_array = x_augmentation_array.astype(np.uint8)

np.save(OUT_DIR + '/' + 'x_augmentation_array_mixup_baseline_nonbeta', x_augmentation_array) #LEILAEDIT for .npy saving
np.save(OUT_DIR + '/' + 'y_augmentation_array_mixup_baseline_nonbeta', y_augmentation_array) #LEILAEDIT for .npy saving
