# web-app for API image manipulation
from __future__ import print_function

import io
from pprint import pprint
from PIL import Image, ImageDraw, ExifTags, ImageColor
import numpy as np
import cv2
import os,sys,shutil
import glob
from detect_edges_image import CropLayer
import argparse
from flask import Flask, request, render_template, send_from_directory
import torch
import torch.nn as nn
from models.networks import Define_G, Define_D
import torch.optim as optim
from config import Config
import os
import os.path as osp
from torch.utils.data import DataLoader
from torchvision import transforms
from data.regular_dataset import RegularDataset
from data.demo_dataset import DemoDataset
from utils.transforms import create_part
from time import time
import datetime
import torch.backends.cudnn as cudnn
import numpy as np
import subprocess
import shlex
from torchvision import utils
from utils import pose_utils
import torch.nn.functional as F
from utils.warp_image import warped_image
from lib.geometric_matching_multi_gpu import GMM
from torchvision import utils
from PIL import Image
import time
from datetime import datetime
import scipy.misc
import scipy.io as sio
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
#from utils import *
from utils.model_pgn import PGNModel
from utils.utils import decode_labels, inv_preprocess, prepare_label, save, load
from utils.ops import conv2d, max_pool, linear
from utils.image_reader import ImageReader
from utils.image_reader_pgn import ImageReaderPGN
from past.builtins import xrange
DATA_TYPE = ['png','PNG','jpg','JPG']
def makedataset():
    
    path_img = "datasets/CIHP/images/"
   

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
  

N_CLASSES = 20
DATA_DIR = './datasets/CIHP2'
LIST_PATH = './datasets/CIHP2/list/val.txt'
DATA_ID_LIST = './datasets/CIHP2/list/val_id.txt'
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines()) 
RESTORE_FROM = 'checkpoint'


"""Create the model and start the evaluation process."""

# Create queue coordinator.
coord = tf.train.Coordinator()
# Load reader.
with tf.name_scope("create_inputs"):
    reader = ImageReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, None, False, False, False, coord)
    image, label, edge_gt = reader.image, reader.label, reader.edge
    image_rev = tf.reverse(image, tf.stack([1]))
    image_list = reader.image_list

image_batch = tf.stack([image, image_rev])
label_batch = tf.expand_dims(label, dim=0) # Add one batch dimension.
edge_gt_batch = tf.expand_dims(edge_gt, dim=0)
h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))
      
# Create network.
with tf.variable_scope('', reuse=False):
    net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
# parsing net

parsing_out1_050 = net_050.layers['parsing_fc']
parsing_out1_075 = net_075.layers['parsing_fc']
parsing_out1_100 = net_100.layers['parsing_fc']
parsing_out1_125 = net_125.layers['parsing_fc']
parsing_out1_150 = net_150.layers['parsing_fc']
parsing_out1_175 = net_175.layers['parsing_fc']

parsing_out2_050 = net_050.layers['parsing_rf_fc']
parsing_out2_075 = net_075.layers['parsing_rf_fc']
parsing_out2_100 = net_100.layers['parsing_rf_fc']
parsing_out2_125 = net_125.layers['parsing_rf_fc']
parsing_out2_150 = net_150.layers['parsing_rf_fc']
parsing_out2_175 = net_175.layers['parsing_rf_fc']

# edge net
edge_out2_100 = net_100.layers['edge_rf_fc']
edge_out2_125 = net_125.layers['edge_rf_fc']
edge_out2_150 = net_150.layers['edge_rf_fc']
edge_out2_175 = net_175.layers['edge_rf_fc']


# combine resize
parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)
                                        
raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
tail_list = tf.unstack(tail_output, num=20, axis=2)
tail_list_rev = [None] * 20
for xx in xrange(14):
    tail_list_rev[xx] = tail_list[xx]
tail_list_rev[14] = tail_list[15]
tail_list_rev[15] = tail_list[14]
tail_list_rev[16] = tail_list[17]
tail_list_rev[17] = tail_list[16]
tail_list_rev[18] = tail_list[19]
tail_list_rev[19] = tail_list[18]
tail_output_rev = tf.stack(tail_list_rev, axis=2)
tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
raw_output_all = tf.expand_dims(raw_output_all, dim=0)
pred_scores = tf.reduce_max(raw_output_all, axis=3)
raw_output_all = tf.argmax(raw_output_all, axis=3)
pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.


raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
pred_edge = tf.sigmoid(raw_edge_all)
res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

# prepare ground truth 
preds = tf.reshape(pred_all, [-1,])
gt = tf.reshape(label_batch, [-1,])
weights = tf.cast(tf.less_equal(gt, N_CLASSES - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
mIoU, update_op_iou = tf.contrib.metrics.streaming_mean_iou(preds, gt, num_classes=N_CLASSES, weights=weights)
macc, update_op_acc = tf.contrib.metrics.streaming_accuracy(preds, gt, weights=weights)

# precision and recall
recall, update_op_recall = tf.contrib.metrics.streaming_recall(res_edge, edge_gt_batch)
precision, update_op_precision = tf.contrib.metrics.streaming_precision(res_edge, edge_gt_batch)

update_op = tf.group(update_op_iou, update_op_acc, update_op_recall, update_op_precision)

# Which variables to load.
restore_var = tf.global_variables()
# Set up tf session and initialize variables. 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)
sess.run(tf.local_variables_initializer())

# Load weights.
loader = tf.train.Saver(var_list=restore_var)
if RESTORE_FROM is not None:
    if load(loader, sess, RESTORE_FROM):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

# Start queue threads.
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

DATA_TYPE = ['png','PNG','jpg','JPG']

print("[INFO] loading edge detector...")
protoPath = os.path.sep.join(["hed_model",
            "deploy.prototxt"])
modelPath = os.path.sep.join(["hed_model",
            "hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

"""
Forward function for vitural try-on
Note : 
      Set opt.istest == True for arbitrary pose and given image
      Set istrain = False and opt.istest == False for validating data in the validation dataset in end2end manner
"""
resume_gmm = "pretrained_checkpoint/step_009000.pth"
resume_G_parse = 'pretrained_checkpoint/parsing.tar'
resume_G_app_cpvton = 'pretrained_checkpoint/app.tar'
resume_G_face = 'pretrained_checkpoint/face.tar'

paths = [resume_gmm, resume_G_parse, resume_G_app_cpvton, resume_G_face]
opt = Config().parse()
if not os.path.exists(opt.forward_save_path):
    os.makedirs(opt.forward_save_path)
refine_path = opt.forward_save_path
def load_model(model, path):

    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint.state_dict())
    model = model.cuda()

    model.eval()
    print(20*'=')
    for param in model.parameters():
        param.requires_grad = False

cudnn.enabled = True
cudnn.benchmark = True
opt.output_nc = 3

gmm = GMM(opt)
gmm = torch.nn.DataParallel(gmm).cuda()

# 'batch'
generator_parsing = Define_G(opt.input_nc_G_parsing, opt.output_nc_parsing, opt.ndf, opt.netG_parsing, opt.norm, 
                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

generator_app_cpvton = Define_G(opt.input_nc_G_app, opt.output_nc_app, opt.ndf, opt.netG_app, opt.norm, 
                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, with_tanh=False)

generator_face = Define_G(opt.input_nc_D_face, opt.output_nc_face, opt.ndf, opt.netG_face, opt.norm, 
                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

models = [gmm, generator_parsing, generator_app_cpvton, generator_face]
for model, path in zip(models, paths):
    load_model(model, path)    
print('==>loaded model')






application = app = Flask(__name__)
APP_ROOT = os.path.basename('.')
# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT)

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".jpeg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    upload.save("static/images/temp.jpg")    
    
    im = Image.open("static/images/temp.jpg")
    if im.mode in ("RGBA", "P"):
      im = im.convert("RGB")
    new_width = 192
    new_height = 256
    im = im.resize((new_width,new_height),Image.ANTIALIAS)
    im.save("dataset/cloth_image/dress.jpg")
     # load our serialized edge detector from disk
  

    # load the input image and grab its dimensions
    image = cv2.imread("dataset/cloth_image/dress.jpg")

    (H, W) = image.shape[:2]



    # construct a blob out of the input image for the Holistically-Nested
    # Edge Detector
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False)

    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    # show the output edge detection results for Canny and
    # Holistically-Nested Edge Detection
    '''cv2.imshow("Input", image)
    cv2.imshow("Canny", canny)
    cv2.imshow("HED", hed)'''

    cv2.imwrite("dataset/cloth_mask/dress_mask.png",hed)
    
    augment = {}

    if '0.4' in torch.__version__:
        augment['3'] = transforms.Compose([
                                    # transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ]) # change to [C, H, W]
        augment['1'] = augment['3']

    else:
        augment['3'] = transforms.Compose([
                                # transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]) # change to [C, H, W]

        augment['1'] = transforms.Compose([
                                # transforms.Resize(256),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
        ]) # change to [C, H, W]
    
    
    val_dataset = DemoDataset(opt, augment=augment)
    val_dataloader = DataLoader(
                    val_dataset,
                    shuffle=False,
                    drop_last=False,
                    num_workers=opt.num_workers,
                    batch_size = opt.batch_size_v,
                    pin_memory=True)
    
    with torch.no_grad():
        for i, result in enumerate(val_dataloader):
            'warped cloth'
            warped_cloth = warped_image(gmm, result) 
            if opt.warp_cloth:
                warped_cloth_name = result['warped_cloth_name']
                warped_cloth_path = os.path.join('dataset', 'warped_cloth', warped_cloth_name[0])
                if not os.path.exists(os.path.split(warped_cloth_path)[0]):
                    os.makedirs(os.path.split(warped_cloth_path)[0])
                utils.save_image(warped_cloth * 0.5 + 0.5, warped_cloth_path)
                print('processing_%d'%i)
                continue 
            source_parse = result['source_parse'].float().cuda()
            target_pose_embedding = result['target_pose_embedding'].float().cuda()
            source_image = result['source_image'].float().cuda()
            cloth_parse = result['cloth_parse'].cuda()
            cloth_image = result['cloth_image'].cuda()
            target_pose_img = result['target_pose_img'].float().cuda()
            cloth_parse = result['cloth_parse'].float().cuda()
            source_parse_vis = result['source_parse_vis'].float().cuda()

            "filter add cloth infomation"
            real_s = source_parse   
            index = [x for x in list(range(20)) if x != 5 and x != 6 and x != 7]
            real_s_ = torch.index_select(real_s, 1, torch.tensor(index).cuda())
            input_parse = torch.cat((real_s_, target_pose_embedding, cloth_parse), 1).cuda()
            
            'P'
            generate_parse = generator_parsing(input_parse) # tanh
            generate_parse = F.softmax(generate_parse, dim=1)

            generate_parse_argmax = torch.argmax(generate_parse, dim=1, keepdim=True).float()
            res = []
            for index in range(20):
                res.append(generate_parse_argmax == index)
            generate_parse_argmax = torch.cat(res, dim=1).float()

            "A"
            image_without_cloth = create_part(source_image, source_parse, 'image_without_cloth', False)
            input_app = torch.cat((image_without_cloth , warped_cloth, generate_parse), 1).cuda()
            generate_img = generator_app_cpvton(input_app)
            p_rendered, m_composite = torch.split(generate_img, 3, 1) 
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
            refine_img = p_tryon

            "F"
            generate_face = create_part(refine_img, generate_parse_argmax, 'face', False)
         
          
            source_face = create_part(source_image, generate_parse_argmax, 'face', False)
            source_face_new = create_part(source_image, source_parse, 'face', False)
            input_face = torch.cat((source_face_new, generate_face), 1)
            fake_face = generator_face(input_face)
            fake_face = create_part(fake_face, generate_parse_argmax, 'face', False) 
            generate_img_without_face = refine_img - generate_face
                      
            refine_img =fake_face + generate_img_without_face
            "generate parse vis"
            if opt.save_time:
                generate_parse_vis = source_parse_vis
            else:
                generate_parse_vis = torch.argmax(generate_parse, dim=1, keepdim=True).permute(0,2,3,1).contiguous()
                generate_parse_vis = pose_utils.decode_labels(generate_parse_vis)
            "save results"
            images = [source_image, cloth_image, refine_img]
            pose_utils.save_img(images, os.path.join(refine_path, '%d.jpg')%(i))

    torch.cuda.empty_cache()
 
       
    #cv2.imwrite("static/images/temp.jpg", image_new)

    


    return send_image('0.jpg')

@app.route("/match", methods=["POST"])
def process():
    target = os.path.join(APP_ROOT)

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("image")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".jpeg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    upload.save("static/images/temp_person_image.jpg")  
    
    im = Image.open("static/images/temp_person_image.jpg")
    if im.mode in ("RGBA", "P"):
      im = im.convert("RGB")
  
    im.save("datasets/CIHP/images/person.jpg")
    makedataset()
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
      

    res_mIou = mIoU.eval(session=sess)
    res_macc = macc.eval(session=sess)
    res_recall = recall.eval(session=sess)
    res_precision = precision.eval(session=sess)
    f1 = 2 * res_precision * res_recall / (res_precision + res_recall)
    print('Mean IoU: {:.4f}, Mean Acc: {:.4f}'.format(res_mIou, res_macc))
    print('Recall: {:.4f}, Precision: {:.4f}, F1 score: {:.4f}'.format(res_recall, res_precision, f1))

    coord.request_stop()
    coord.join(threads)   
    return render_template('processing.html')



# retrieve file from 'static/images' directory
@app.route('/end2end/<filename>')
def send_image(filename):
    return send_from_directory("end2end", filename)


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=80)

