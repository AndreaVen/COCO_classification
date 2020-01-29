# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:06:09 2020

@author:Andrea Venditti
this script uses a portion of the coco dataset (train or val) and uses this to generate a small dataset for a classification task
all the images of the relative classes (specified in the lista_tag variable) will be put together in the dataALL folder. this script
must be called from the dict_generator, the first time the script runs the flag_generate must be set to 1 and the dataDir must be set to the
folder where the COCO data is
"""
def genera_coco(flag_generate=0,dataDir='R:\\coco_dataset')
    from pycocotools.coco import COCO
    import numpy as np
    import skimage.io as io
    import matplotlib.pyplot as plt
    import os 
    import time
    from matplotlib import pylab
    import cv2
    from tqdm import *
    from pylab import *
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    dataType='train2014'
    annFile='{}\\annotations\\instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    coco.anns
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    nms = set([cat['supercategory'] for cat in cats])
    #list of tag of interest
    lista_tag=['person','car','airplane','train','bird','cat','dog','broccoli','vase','bottle']
    matrix=[]
    
    for tag in  lista_tag:
        catIds = coco.getCatIds(catNms=tag);
        matrix.append(coco.getImgIds(catIds=catIds))
    lenmat=[]
    for i in (matrix):
        lenmat.append(len(i))
        
        
        # random crop of the image 
    def randomCrop(img, mask, width, height):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        x = np.random.randint(0, img.shape[1] - width)
        y = np.random.randint(0, img.shape[0] - height)
        img = img[y:y+int(height*0.8), x:x+int(width*0.8)]
        mask = mask[y:y+int(height*0.8), x:x+int(width*0.8)]
        return  mask
        
    toremove=[] 
    for i in range(len(matrix)-1):
        for k in range (i+1,len(matrix)):
            tmp=list(set(matrix[i])&set(matrix[k]))#get common names and add them to a list to be removed
            for jj in tmp:
                toremove.append(jj)   
                
    for ii in toremove: #make mutually exclusive classes
        for hh in range(len(matrix)):
            if ii in matrix[hh]:
                matrix[hh].remove(ii)
    
    def reduce_matrix(matrix,max=100):# function to limit the maximum number of examples for every class, this reduces the unbalance
        for i in matrix:
            while len(i)>max:
                i.remove(i[np.random.randint(0,len(i))])
    maxx=1000# maximum number of examples per class
    reduce_matrix(matrix,maxx)         
            
    
    vector_decrease=[]
    lenmatFin=[]
    for i in (matrix):
        lenmatFin.append(len(i))       
    
    decrease=1-(sum(lenmat)-sum(lenmatFin))/sum(lenmat) #decrease of number of samples
    
    
    
    #order classes, from higher number of examples to lower 
    var1=np.array(lenmatFin)
    var1=(-var1).argsort()
    matrixO=[]
    lista_tagO=[]
    for i in var1:
        matrixO.append(matrix[i])
        lista_tagO.append(lista_tag[i])
        
    
    #keep track of the difference in number of examples 
    diff=[]
    for jj in matrixO:
        diff.append((len(matrixO[0])-len(jj)))
    
    
    
            
    def generatore_esclusivo():         
        if not os.path.exists(dataDir+'\\'+'dataALL_10class'):
            os.mkdir(dataDir+'\\'+'dataALL_10class')
        for ii in range(len(matrixO)):
            for kk in tqdm( matrixO[ii]):
                img = coco.loadImgs(kk)[0]
                I = io.imread('%s\\%s\\%s'%(dataDir,dataType,img['file_name']))
                percorso=dataDir+'\\'+'dataALL_10class'+'\\'+lista_tagO[ii]+''-'+str(kk)+'.jpg'
                io.imsave(percorso,I) 
                while diff[ii]>0:
                    img = coco.loadImgs(matrixO[ii][np.random.randint(0,len(matrixO[ii]))])[0]
                    I = io.imread('%s\\%s\\%s'%(dataDir,dataType,img['file_name']))
                    mask=I
                    tmp=randomCrop(I,mask,int(I.shape[1]*0.9),int(I.shape[0]*0.9))
                    tmp=cv2.flip(tmp,np.random.randint(-1,2)) #
                    percorso=dataDir+'\\'+'dataALL_10class'+'\\'+lista_tagO[ii]+'-'+str(kk)+'aug'+str(diff[ii])+'.jpg'
                    io.imsave(percorso,tmp)
                    diff[ii]-=1
               
    if flag_generate==1:
        generatore_esclusivo()
    
        
    
    
    def showimg(): #function to show an image, debug only
        catIds = coco.getCatIds(catNms=['person']); 
        imgIds = coco.getImgIds(catIds=catIds );
        img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    
        I = io.imread('%s\\%s\\%s'%(dataDir,dataType,img['file_name']))
        plt.axis('off')
        plt.imshow(I)
        plt.show()
    
    def check_categoria(num=421566): #function to check the tag of BB in an image, debuf only
        annotation_ids = coco.getAnnIds(num)
        annotations = coco.loadAnns(annotation_ids)
    
        for i in range(len(annotations)):
            entity_id = annotations[i]["category_id"]
            entity = coco.loadCats(entity_id)[0]["name"]
            print("{}: {}".format(i, entity))
        for kk in range(len(matrix)):
            if num in matrix[kk]:
                print("image in",lista_tag[kk])
    
    return lista_tagO,maxx 
