# This file trains a ResNet-110, with a user-specified data augmentation set that is added to the original training set.
# It is based off a script implemented by Markus Kangsepp
# ResNet model from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pickle

# Constants
stack_n            = 18            
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 49500 // batch_size # Edit the numerator to equal the total augmented training set size (45,000 + aug_set_size)
weight_decay       = 0.0001

seed = 333

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 150:
        return 0.01
    return 0.001

# Define the ResNet
def residual_network(img_input,classes_num=10,stack_n=5):
    
    # Define residual blocks
    def residual_block(intput,out_channel,increase=False):
        if increase:
            stride = (2,2)
        else:
            stride = (1,1)

        pre_bn   = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1   = BatchNormalization()(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput,conv_2])
        return block

    # total layers = stack_n * 3 * 2 + 2
    # stack_n = 5 by default, total layers = 32
    # Input dimensions: 32x32x3 
    # Output dimensions: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # Input dimensions: 32x32x16
    # Output dimensions: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # Input dimensions: 32x32x16
    # Output dimensions: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
    # Input dimensions: 16x16x32
    # Output dimensions: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes_num,activation='softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

if __name__ == '__main__':
    
    # Load in the CIFAR-10 test set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_test = keras.utils.to_categorical(y_test, num_classes10)
    
    # Load in the augmentation set for the experiment being run. This file should be adjusted for each distinct experiment.
    x_train_additions = np.load('Augmentation_Sets/x_augmentation_array_mean_beta_largesample_4500.npy')
    y_train_additions = np.load('Augmentation_Sets/y_augmentation_array_mean_beta_largesample_4500.npy')

    # Split the CIFAR-10 dataset into training, validation, and test sets before appending the augmentation set
    # The random seed ensures that every discriminator uses the same split
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed) 
    
    # Add the augmentation set to the training set
    x_train_additions = x_train_additions.transpose(0,2,3,1)
    y_train45 = keras.utils.to_categorical(y_train45, num_classes10)
    y_train_additions = y_train_additions.reshape(-1, num_classes10)
    x_train45 = np.concatenate((x_train45, x_train_additions),axis=0)
    y_train45 = np.concatenate((y_train45, y_train_additions),axis=0)
    
    # Pre-process colors, as specified in the paper
    img_mean = x_train45.mean(axis=0)  # per-pixel mean
    img_std = x_train45.std(axis=0)
    x_train45 = (x_train45-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std
    
    # Assemble the neural network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output)
    print(resnet.summary())

    # Set the optimizer and momentum, specify gradient clipping value with the clipnorm option to ensure stability
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True, clipnorm=1.)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Set the callback
    cbks = [LearningRateScheduler(scheduler)]

    # Simple data augmentation (e.g., flips) - these do not change the class vectors
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)
    datagen.fit(x_train45)

    # Commence training
    hist = resnet.fit_generator(datagen.flow(x_train45, y_train45,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_val, y_val))
    
    # Save model weights once training finishes
    resnet.save('resnet_110_45kclip_augmean_beta_largesample_4500.h5')
    
    print("Get test accuracy:")
    loss, accuracy = resnet.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    # Save model history
    with open('hist_110_cifar10_v2_45kclip_augmean_beta_largesample_4500.p', 'wb') as f:
        pickle.dump(hist.history, f)
