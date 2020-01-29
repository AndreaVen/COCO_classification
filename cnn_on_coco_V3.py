# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:58:35 2020

@author: Andrea Venditti

"""
# -*- coding: utf-8 
import keras
from keras.models import Model ,Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt 
import cv2
import os
from keras import applications
from random import shuffle 
from tqdm import tqdm
import tensorflow as tf 
from keras.optimizers import *
from keras.utils import np_utils
from keras_data_generator import DataGenerator
import matplotlib.pyplot as plt
import time
from dict_generator import generate_dict
img_width=img_height=224

from keras.models import Sequential
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
path='R:\\coco_dataset'
path_to_save=path+'\\dati_salvati'
# Parameters
# Datasets
partition,labels = generate_dict(0,path)
if __name__ == '__main__':
    params = {'dim': (224,224),
          'batch_size': 16,
          'n_classes': 10,
          'n_channels': 3,'shuffle': True}
    maxscore=0
    d1S=[2048  ]
    d2S=[2048 ]
    d3S=[0]
    llrS=[ 0.00001]
    dr1S=[0.2 ]
    dr2S=[0.2]
    layerIDXS=[1] # -1= all the layers 
    for d1 in d1S:
        for d2 in d2S:
            for d3 in d3S:
                for llr in llrS:
                    for dr1 in dr1S:
                        for dr2 in dr2S:
                            K.clear_session()
                            model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
                            for layerIDX in  layerIDXS:   
                                    
                                for layerr in model.layers[0:layerIDX]: #19 layer totali 
                                    layerr.trainable = False
                           
                                x = model.output
                                x = Flatten()(x)
                                x = Dense(d1, activation="relu")(x)
                                x = Dropout(dr1)(x)
                                x = Dense(d2, activation="relu")(x)
                                if d3!=0: 
                                    x = Dropout(dr2)(x)
                                    x = Dense(d3, activation="relu")(x)
                                
                                predictions = Dense(10, activation="softmax")(x)
                                
                                training_generator = DataGenerator(partition['train'], labels, **params)
                                validation_generator = DataGenerator(partition['validation'], labels, **params)
                                test_generator=DataGenerator(partition['test'], labels,**params)
                                
                                # creating the final model 
                                model_final = Model(input = model.input, output = predictions)
                                del model
                                #model_final.summary()
                                model_final.summary()
                                # model_final.load_weights("R:\\coco_dataset\\dati_salvati\\coco_vgg16_generator1.h5")
                                # compile the model 
                                model_final.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(llr, momentum=0.9), metrics=["accuracy"])
                                # Initiate the train and test generators with data Augumentation 
                                NAME='d1_{}_dr1_{}-d2_{}_dr2_{}-d3_{}_lr_{}_layeridx_{}-{}'.format(d1,dr1,d2,dr2,d3,llr,layerIDX,int(time.time()))
                                tensorboard=TensorBoard(log_dir='./logs/{}'.format(NAME))
                                # Save the model according to the conditions  
                                check_name=path_to_save+'\\coco_vgg16_generatord1_{}_dr1_{}-d2_{}_dr2_{}-d3_{}_lr_{}_layeridx_{}-{}'.format(d1,dr1,d2,dr2,d3,llr,layerIDX,int(time.time()))
                                checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                                
                                early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
                                model_final.fit_generator(generator=training_generator,
                                                    validation_data=validation_generator,
                                                    use_multiprocessing=False,
                                                    workers=16,epochs=40,callbacks = [early,tensorboard,checkpoint])
                                
                                model_final.load_weights(check_name)
                                score=model_final.evaluate_generator(generator=test_generator)
                                if score[1]>maxscore:
                                    model_name=path_to_save+'\\coco_vgg16_best_d1_{}_dr1_{}-d2_{}_dr2_{}-d3_{}_lr_{}_score_{}-{}.h5'.format(d1,dr1,d2,dr2,d3,llr,score[1],int(time.time()))
                                    model_final.save(model_name)
                                
                                # del model_final
                                K.clear_session()
                                tf.reset_default_graph()
                                iii=keras.model.load()
                              
                               

model_final=load_model(model_name) #load pretrained network with weights 

# score = model_final.evaluate_generator(generator=test_generator, use_multiprocessing=False, workers=16)


