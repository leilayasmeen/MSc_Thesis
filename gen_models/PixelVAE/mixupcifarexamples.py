import os, sys
sys.path.append(os.getcwd())

OUT_DIR = 'mixup_cifar_examples'

import numpy as np
import tensorflow as tf
import imageio
from imageio import imsave
import random

import keras

import time
import functools

import sklearn
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
N_CHANNELS = 3
HEIGHT = 32
WIDTH = 32

if not os.path.isdir(OUT_DIR):
   os.makedirs(OUT_DIR)
   print "Created directory {}".format(OUT_DIR)
      
from keras.datasets import cifar10
(x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
   
x_train_set = x_train_set.transpose(0,3,1,2)
x_test_set = x_test_set.transpose(0,3,1,2)
    
seed = 333
x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)

from keras.utils import np_utils           
            
# Function to translate numeric images into plots
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
                
numsamples = 1
pvals = np.linspace(0.0, 1.0, num=6)
                
print "Sampling First Five Image Indices from Each Class"
for k in range(NUM_CLASSES-1):
      idk = np.asarray(np.where(np.equal(y_train_set,k))[0])
      x_classk = np.asarray(x_train_set[idk,:])
      idx = random.sample(range(0, x_classk.shape[0]), 1)
                           
      print "Drawing Corresponding Image and Label Out"            
      x_classk_array = np.array(x_classk)
      image1 = x_classk[idx,:]

      # Reshape
      image1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
      
      for j in range(k+1,NUM_CLASSES):
            idj = np.asarray(np.where(np.equal(y_train_set,j))[0])
            x_classj = np.asarray(x_train_set[idj,:])
            idx2 = random.sample(range(0, x_classj.shape[0]), 1)
                           
            print "Drawing Corresponding Image and Label Out"            
            image2 = x_classj[idx2,:]

            # Reshape
            image2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
            print "Saving mixed examples"
            
            for p in pvals:
                  sample = p*image1 + (1.0-p)*image2
                  color_grid_vis(
                        sample,
                        1,
                        1,
                        'mixup_class{}_andclass{}_pvaluefirstclass{}.png'.format(k,j,p)
                  )
