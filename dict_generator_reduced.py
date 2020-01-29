# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:20:52 2020

@author: andrea
"""
def generate_dict(generate_flag=0, path='R:\\coco_dataset',relevant_tag=['person','dog','cat']):
   
    import os 
    path_to_save=os.path.join(path,'dati_salvati')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    if generate_flag==1:
        from dict_generator import generate_dict
        
        partition,labels = generate_dict(0,Path)
        label_reduced={}
        data_reduced={'train':[],'validation':[],'test':[]}
        
        
        
        
        
        for i in partition['train']:
           
            tmp=os.path.basename(i).split('-')[0]
            if tmp in relevant_tag:
                data_reduced['train'].append(i)
                label_reduced[tmp]=labels[i]
            
        for i in partition['test']:
            tmp=os.path.basename(i).split('-')[0]
            if tmp in relevant_tag:
                data_reduced['test'].append(i)
                label_reduced[tmp]=labels[i]
                
        for i in partition['validation']:
            tmp=os.path.basename(i).split('-')[0]
            if tmp in relevant_tag:
                data_reduced['validation'].append(i)
                label_reduced[tmp]=labels[i]
                #when removing some classes it is necessary to modify the label index as well as there will be otherwise
        #some problem when one hot labeling inside the keras generator
        lista_num=[]
        for i in label_reduced:
            if label_reduced[i][0] not in lista_num:
                lista_num.append(label_reduced[i][0])
        
        min_num=0
        for k in lista_num:
            for i in label_reduced:
                if label_reduced[i][0]==k:
                    label_reduced[i][0]=min_num
            min_num+=1
        
        
        
        f = open(path_to_save+'\\data_reduced.txt',"w")
        f.write( str(data_reduced) )
        f.close()   
        f = open(path_to_save+'\\label_reduced.txt',"w")
        f.write( str(label_reduced) )
        f.close() 
    
    
    else:
        with open(path_to_save+'\\data_reduced.txt','r') as inf:
            data_reduced = eval(inf.read())
        with open(path_to_save+'\\label_reduced.txt','r') as inf:
            label_reduced = eval(inf.read())
    


    return data_reduced,label_reduced
