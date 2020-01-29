# COCO_classification
The COCO classification is an object detection dataset. The objective of this repository is to use it for an image classification task.
How to use:
download the validation or train sets from http://cocodataset.org/#download
extract che dataset and the annotation so thath the directory structure is 
    │
    ├─Val2014
├─annotations
│ ├─istances_val2014
├─dataALL_10class
├─Dati_salvati

The cnn_on_cocov3 calls the dict_generator which in turn calls the genera_coco_classificationV2
genera_coco_classificationV2 takes the list of class and saves them in a folder making sure they are mutually exclusive
dict_generator takes the images and builds a dictionary of full path divided in trian test val and a dictionary of labels
cnn_on_coco gets the dictionary and feed it to the keras_data_generator, using this it perform a transfer learning using VGG16
The "reduced" and "other" version load the pretrained network and use it on a simplified problem
