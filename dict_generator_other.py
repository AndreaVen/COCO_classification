# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:47:30 2020

@author: andrea
"""

import numpy as np
import skimage.io as io
import os
def generate_dict(generate_flag=0,path='R:\\coco_dataset',relevant_tag=['person','dog','cat']):
    
    path_to_save=os.path.join(path,'dati_salvati')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if generate_flag==1:
        from dict_generator import generate_dict
        partition,labels = generate_dict(0)
        num=np.zeros((10,1))
        label_other={}
        data_other={'train':[],'validation':[],'test':[]}
        for i in partition['train']:
            if labels[i][1] in relevant_tag:
                data_other['train'].append(i)
                label_other[i]=[labels[i][0],labels[i][1]] # relevant_tag classes are copy as is
            elif (int(num[labels[i][0]])<=1050/7): # I want 1/7 of all the training set per class
                data_other['train'].append(i)
                label_other[i]=[99,'other'] # the number of the class will be changed in another script and the class is always other 
                num[labels[i][0]]+=1
                
        num=np.zeros((10,1))   
        for i in partition['validation']:
            if labels[i][1] in relevant_tag:
                data_other['validation'].append(i)
                label_other[i]=[labels[i][0],labels[i][1]] # relevant classes are copy as is
            elif (int(num[labels[i][0]])<=150/7): # I want 1/7 of all the validationing set per class
                data_other['validation'].append(i)
                label_other[i]=[99,'other'] # the number of the class will be changed in another script and the class is always other 
                num[labels[i][0]]+=1
            
     
            
        num=np.zeros((10,1))
        for i in partition['test']:
             if labels[i][1] in relevant_tag:
                 data_other['test'].append(i)
                 label_other[i]=[labels[i][0],labels[i][1]] # relevant classes are copy as is
             elif (int(num[labels[i][0]])<=300/7): # I want 1/7 of all the testing set per class
                 data_other['test'].append(i)
                 label_other[i]=[99,'other'] # the number of the class will be changed in another script and the class is always other 
                 num[labels[i][0]]+=1
            
            
        #when removing some classes it is necessary to modify the label index as well as there will be otherwise
        #some problem when one hot labeling inside the keras generator
        lista_num=[]
        for i in label_other:
            if label_other[i][0] not in lista_num:
                lista_num.append(label_other[i][0])
        
        min_num=0
        for k in lista_num:
            for i in label_other:
                if label_other[i][0]==k:
                    label_other[i][0]=min_num
            min_num+=1
        
        
        
        f = open(path_to_save+'\\data_other.txt',"w")
        f.write( str(data_other) )
        f.close()   
        f = open(path_to_save+'\\label_other.txt',"w")
        f.write( str(label_other) )
        f.close() 
    
    
    else:
        with open(path_to_save+'\\data_other.txt','r') as inf:
            data_other = eval(inf.read())
        with open(path_to_save+'\\label_other.txt','r') as inf:
            label_other = eval(inf.read())
    


    return data_other,label_other
