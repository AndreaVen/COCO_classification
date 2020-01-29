# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:54:03 2020

@author: Andrea Venditti
this script calls the  genera_coco_classificationV2, it will load all the images in the folder and generate a dictionary 
containing the name (path) and label of all images, this will be used to feed a keras generator. the split proportion between
test, validation adn training set can be modified from the split_proportion variable
"""
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
from matplotlib import pylab
import cv2
from tqdm import *
import json
import random





# data_distribution={'classes':[],'number':[]}
# for i in tqdm(os.listdir(path)):
#     tmp=i.split('-')[0] # get the class name
#     if tmp not in data_distribution['classes']:
#          data_distribution['classes'].append(tmp)
#          data_distribution['number'].append(1)
#     else:
#         index= data_distribution['classes'].index(tmp)
#         data_distribution['number'][index]+=1
        
  
    
def find_tag(stringa,lista_tag):
    for i in range(len(lista_tag)):
        if lista_tag[i] in stringa:
            return i
            



def generate_dict(generate_flag=0,Path='R:\\coco_dataset'):
    
    path_to_save=Path+'\\dati_salvati'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    split_proportions=[0.7,0.1,0.2] #split of train, validation,test
    path=Path+'\\'+'dataALL_10class'
    if generate_flag:
        file_list =[]#append all the image full path on a list
        import genera_coco_classificationV2 
        lista_tag,num_ex= genera_coco(flag_generate=1,dataDir=Path)
        
    
        for i in tqdm(os.listdir(path)):
            file_list.append(os.path.join(path,i))
        from genera_coco_classificationV2 import return_list 
        lista_tag,num_ex=return_list()
          #train, validatio, test
        print('len num files=',len(file_list))
        random.shuffle(file_list)  
        num=np.zeros((len(lista_tag),1))
        label={}
        data={'train':[],'validation':[],'test':[]}
        for k in range(len(file_list)):             
            tmp=find_tag(file_list[k],lista_tag)
            if int(num[tmp]<split_proportions[0]*num_ex):
               data['train'].append(file_list[k])
    #           ohl=np.zeros((len(lista_tag),1))
               ohl=tmp
               label[file_list[k]]=[ohl,lista_tag[tmp]]
               num[tmp]+=1
            elif int(num[tmp]>split_proportions[0]*num_ex) and int(num[tmp]<=(1-split_proportions[2])*num_ex):
                data['validation'].append(file_list[k])
                ohl=np.zeros((len(lista_tag),1))
                ohl=tmp
                label[file_list[k]]=[ohl,lista_tag[tmp]]
                num[tmp]+=1
            else:
                data['test'].append(file_list[k])
                ohl=np.zeros((len(lista_tag),1))
                ohl=tmp
                label[file_list[k]]=[ohl,lista_tag[tmp]]
                num[tmp]+=1
       #save the tag list and the number of examples per class
        with open(path_to_save+'\\lista_tag_10.txt', 'w') as filehandle:
            json.dump(lista_tag, filehandle)
        with open(path_to_save+'\\num_ex_10.txt', 'w') as filehandle:
            json.dump(num_ex, filehandle)
         #saves the data and label dictionary    
        f = open(path_to_save+'\\data.txt',"w")
        f.write( str(data) )
        f.close()   
        f = open(path_to_save+'\\label.txt',"w")
        f.write( str(label) )
        f.close() 
    else:  # if the flag is 0 the script will load the dictionary from a file, it is STRONGLY ADVISED to only use flag==1 
        #the first time to create the images and save the dictionary
           
        with open(path_to_save+'\\lista_tag_10.txt', 'r') as filehandle:
           lista_tag=json.load(filehandle)
        with open(path_to_save+'\\num_ex_10.txt', 'r') as filehandle:
           num_ex=json.load(filehandle)
        
        #load the data and label dictionary
        with open(path_to_save+'\\data.txt','r') as inf:
            data = eval(inf.read())
        with open(path_to_save+'\\label.txt','r') as inf:
            label = eval(inf.read())
    
      
    return data,label

