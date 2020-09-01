from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
import shutil
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob

import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *
from past.builtins import xrange
import glob
from detect_edges_image import CropLayer
import argparse
from parsing import main

import os
main()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
 # evaluate prosessing
parsing_dir = 'dataset/parse_cihp'
if os.path.exists(parsing_dir):
    shutil.rmtree(parsing_dir)
if not os.path.exists(parsing_dir):
    os.makedirs(parsing_dir)

# Iterate over training steps.
for step in range(NUM_STEPS):
    parsing_, scores, edge_, _ = sess.run([pred_all, pred_scores, pred_edge, update_op])
    if step % 100 == 0:
        print('step {:d}'.format(step))
        print (image_list[step])
    img_split = image_list[step].split('/')
    img_id = img_split[-1][:-4]
    
    msk = decode_labels(parsing_, num_classes=N_CLASSES)
    parsing_im = Image.fromarray(msk[0])
    parsing_im.save('dataset/parse_cihp/person_vis.png')
    im=Image.open('dataset/parse_cihp/person_vis.png')
    new_width = 192
    new_height = 256
    im = im.resize((new_width,new_height),Image.ANTIALIAS)
    im.save('dataset/parse_cihp/person_vis.png')
    

    cv2.imwrite('dataset/parse_cihp/person.png', parsing_[0,:,:,0])
    im=Image.open('dataset/parse_cihp/person.png')
    im = im.resize((new_width,new_height),Image.ANTIALIAS)
    im.save('dataset/parse_cihp/person.png')
      
    
    #sio.savemat('{}/{}.mat'.format(parsing_dir, img_id), {'data': scores[0,:,:]})
    
    #cv2.imwrite('dataset/cloth_mask/person_mask.png', edge_[0,:,:,0] * 255)
  



coord.request_stop()
coord.join(threads)