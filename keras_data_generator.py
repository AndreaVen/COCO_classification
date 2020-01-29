# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:56:45 2020

@author: Andrea Venditti
this script return a keras generator from a dictionary containing the name (path) and label of some portion of the dataset 
"""
import numpy as np
import keras
import cv2
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels,
                 n_classes, shuffle):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            img=cv2.imread(ID).astype('float32')
            img=cv2.resize(img,self.dim,self.dim)
            
            
            img[:,:,0]=img[:,:,0]-103.939 # this is the normalization used in the pre trained VGG16 model, other pre processing of the image goes here
            img[:,:,1]=img[:,:,1]-116.779
            img[:,:,2]=img[:,:,2]-123.68
            
            X[i,]=img
            
#            X[i,] = np.load('data/' + ID + '.npy')





            # Store class
            y[i] = self.labels[ID][0]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)