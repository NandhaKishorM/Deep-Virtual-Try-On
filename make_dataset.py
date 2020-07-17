# dataset maker for https://github.com/Engineering-Course/CIHP_PGN
# author: qzane@live.com

import os,sys,shutil
from PIL import Image
import numpy as np
import cv2
import glob
import argparse

DATA_TYPE = ['png','PNG','jpg','JPG']
def makedataset():
    
    path_img = "datasets/CIHP/images/"
    path_dress =  "datasets/CIHP/dress/"
    path_dress_final = "dataset/cloth_image/"
    if os.path.isdir(path_dress_final):
        shutil.rmtree(path_dress_final)
        #shutil.rmtree(path_dataset)
    os.makedirs(path_dress_final)
    person_image = "dataset/images/"
    if os.path.isdir(person_image):
        shutil.rmtree(person_image)
        #shutil.rmtree(path_dataset)
    os.makedirs(person_image)
    name_dataset = "CIHP2"
   
    path_dataset = os.path.join('.', 'datasets', name_dataset)
    if os.path.isdir(path_dataset):
        shutil.rmtree(path_dataset)
        #shutil.rmtree(path_dataset)
    os.makedirs(path_dataset)
    path_edge = os.path.join(path_dataset, 'edges')
    path_images = os.path.join(path_dataset, 'images')
    
    
    path_list = os.path.join(path_dataset, 'list')
    for p in [path_edge, path_images, path_list]:
        os.makedirs(p)
    files = [i for i in os.listdir(path_img) if i.split('.')[-1] in DATA_TYPE]
    for f in files:
        im = Image.open(os.path.join(path_img, f))
        '''new_width = 640
        new_height = 480
        im = im.resize((new_width,new_height),Image.ANTIALIAS)'''
        im = im.resize((im.size[0]*720//im.size[1],720), Image.LANCZOS) # if you run out of GPU memory
        im1 = Image.new('L', im.size)
        im.save(os.path.join(path_images, f))
        im1.save(os.path.join(path_edge, '.'.join(f.split('.')[:-1])+'.png'))
    files = [i for i in os.listdir(path_dress) if i.split('.')[-1] in DATA_TYPE]
    for f in files:
        im = Image.open(os.path.join(path_dress, f))
        new_width = 192
        new_height = 256
        im = im.resize((new_width,new_height),Image.ANTIALIAS)
        im.save("dataset/cloth_image/dress.jpg")
        '''path_dress_mask = "dataset/cloth_mask/"
        if os.path.isdir( path_dress_mask ):
            shutil.rmtree( path_dress_mask )
       
        os.makedirs( path_dress_mask )'''
       

    files = [i for i in os.listdir(path_img) if i.split('.')[-1] in DATA_TYPE]
    for f in files:
        im = Image.open(os.path.join(path_img , f))
        new_width = 192
        new_height = 256
        im = im.resize((new_width,new_height),Image.ANTIALIAS)
        im.save("dataset/images/person.jpg")
        
    with open(os.path.join(path_list, 'val.txt'), 'w') as flist:
        for f in files:
            flist.write('/images/%s /edges/%s\n'%(f,'.'.join(f.split('.')[:-1])+'.png'))
    with open(os.path.join(path_list, 'val_id.txt'), 'w') as flist:
        for f in files:
            flist.write('%s\n'%'.'.join(f.split('.')[:-1]))
  
        



if __name__ == '__main__':
    makedataset()
