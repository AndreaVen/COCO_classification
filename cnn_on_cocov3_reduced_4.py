# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:59:10 2020

@author: Andrea Venditti
This version removes 7 of the 10 classes to simplify the problem and an extra class "other" is used. The 2 dense layer of the network are re initialized as random
"""

# -*- coding: utf-8 -*-

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
from dict_generator_other import generate_dict
img_width=img_height=224

from keras.models import Sequential
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
path='R:\\coco_dataset'
path_to_save=path+'\\dati_salvati'
# Parameters
# Datasets
partition,labels = generate_dict(0)
if __name__ == '__main__':
    params = {'dim': (224,224),
          'batch_size': 16,
          'n_classes': 4,
          'n_channels': 3,'shuffle': True}
    maxscore=0
    d1=[2048  ]
    d2=[2048 ]
    d3=[0]
    llr=[ 0.00001]
    dr1=[0.2 ]
    dr2=[0.2]
    
    K.clear_session()
    model=load_model('R:\\coco_dataset\\dati_salvati\\coco_vgg16_best_d1_2048_dr1_0.2-d2_2048_dr2_0.2-d3_0_lr_1e-05_score_0.7954545454545454-1580134149.h5') #load pretrained network with weights 
    model.layers.pop() #remove the last layer that is dependent on classes
    x = model.layers[-1].output
    predictions = Dense(4, activation="softmax",name='softmax_layer')(x)
    
    # layerIDX=5 #final_model has 23 layer i.e from 0 to 22
    # for layerr in model.layers[0:layerIDX]: #19 layer totali 
    #      layerr.trainable = False
        
    
    
    # creating the final model 
    model_final = Model(input = model.input, output = predictions)
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
    test_generator=DataGenerator(partition['test'], labels,**params)
    del model
    model_final.summary()
 
    # model_final.load_weights("R:\\coco_dataset\\dati_salvati\\coco_vgg16_generator1.h5")
    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(0.00001, momentum=0.9), metrics=["accuracy"])
    # Initiate the train and test generators with data Augumentation 
    NAME='Reduced_4_{}'.format(int(time.time()))
    tensorboard=TensorBoard(log_dir='./logs/{}'.format(NAME))
    # Save the model according to the conditions  
    check_name=path_to_save+'\\coco_vgg16_reduced_4_{}'.format(int(time.time()))
    checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=8, verbose=1, mode='auto')
    model_final.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=16,epochs=40,callbacks = [early,tensorboard,checkpoint])
    
    model_final.load_weights(check_name)
    score=model_final.evaluate_generator(generator=test_generator)
    # if score[1]>maxscore:
    #     model_final.save('R:\\coco_dataset\\dati_salvati\\coco_vgg16_best_reduced-{}'.format(int(time.time())))
    
    # del model_final
    # K.clear_session()
    # tf.reset_default_graph()
    
  
                               



# score = model_final.evaluate_generator(generator=test_generator, use_multiprocessing=False, workers=16)


# img=cv2.imread(partition['validation'][34])

# plt.imshow(img)
# plt.Annotation(labels['validation'][34])


# model_final=load_model('R:\\coco_dataset\\dati_salvati\\coco_vgg16_best')


# def reset_keras():
#     from keras.backend.tensorflow_backend import set_session
#     from keras.backend.tensorflow_backend import clear_session
#     from keras.backend.tensorflow_backend import get_session
#     sess = get_session()
#     clear_session()
#     sess.close()
#     sess = get_session()

#     try:
#         # del classifier # this is from global space - change this as you need
#     except:
#         pass

#     # print(gc.collect()) # if it's done something you should see a number being outputted

#     # use the same config as you used to create the session
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 1
#     config.gpu_options.visible_device_list = "0"
#     set_session(tf.Session(config=config))

# reset_keras()
