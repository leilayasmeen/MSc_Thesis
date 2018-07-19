import os, sys
sys.path.append(os.getcwd())

OUT_DIR = 'mixup_cifar_examples'

import numpy as np
import tensorflow as tf
import imageio
from imageio import imsave

import keras

import time
import functools

import sklearn
from sklearn.model_selection import train_test_split

NUM_CLASSES = 10
      
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
                
            numsamples = 5
                
            for imagenum in range(numsamples):

                print "Sampling Random Image Index from Each Class"
                for k in range(NUM_CLASSES):
                  idk = np.asarray(np.where(np.equal(y_train_set,k))[0])
                  x_classk = np.asarray(x_train_set[idk,:])
                  #idx = random.sample(range(0, x_classk.shape[0]), 1)
                  idx = imagenum
                           
                  print "Drawing Corresponding Image and Label Out"            
                  #x_train_set_array = np.array(x_train_set)
                  #image = x_train_set_array[idx,:]
                  x_classk_array = np.array(x_classk)
                  image = x_classk[idx,:]

                  # Reshape
                  image = image.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
                  
                  # Encode the images
                  image_code = enc_fn(image)
               
                  samples = np.zeros(
                     (1, N_CHANNELS, HEIGHT, WIDTH), 
                     dtype='int32'
                  )

                  print "Generating samples"
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                           for ch in xrange(N_CHANNELS):
                              next_sample = dec1_fn(image_code, samples, ch, y, x) 
                              samples[:,ch,y,x] = next_sample
                            
                  #LEILAEDIT for .npy saving
                  x_augmentation_set = np.concatenate((x_augmentation_set, samples), axis=0)#LEILAEDIT for .npy saving
                
                  print "Saving original sample"
                  color_grid_vis(
                     image, 
                     1, 
                     1, 
                     'original_class{}_{}.png'.format(k,imagenum)
                  )
                  print "Saving reconstructed sample"
                  color_grid_vis(
                     samples, 
                     1, 
                     1, 
                     'reconstruction_filter_5_class{}_{}.png'.format(k,imagenum)
                  )
