import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--testrun', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32
lr = 1e-3

# 
dirpath = os.getcwd()
if not os.path.isdir(dirpath + "/val_result"):
    os.mkdir(dirpath + "/val_result")
if not os.path.isdir(dirpath + "/train_result"):
    os.mkdir(dirpath + "/train_result")
if not os.path.isdir(dirpath + "/test_result"):
    os.mkdir(dirpath + "/test_result")


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

def power_print(msg, fp):
    print(msg)
    fp.write(msg + "\n")
    fp.flush()


print('--------- SETTING UP ----------')
print(f'args: {args}')


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320, testrun=args.testrun)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320, testrun=args.testrun)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr=lr)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    train_losses = []  # accu train loss over epochs
    val_losses = []  # accu val loss over epochs
    train_size = len(dataloader)

    logfile = open(f'train_{datetime.now().strftime("%m%d_%H_%M")}.log',  'w')
    power_print(datetime.now().strftime("%Y/%m/%d %H:%M"), logfile)

    for epoch in range(num_epochs):
        power_print(f'---------- EPOCH {epoch + 1} ------------', logfile)
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        power_print(f'...training', logfile)
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, img_names, img_height, img_width = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
            # if (i+1) % 100 == 0:
            print(f'...image {i+1}/{train_size}')

        train_losses.append((avg_loss/avg_count).cpu().numpy())
        power_print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count), logfile)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
        windowname = f'train_result/train_{epoch + 1}_{img_names[0]}'
        visualize_pred(windowname, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #VALIDATION
        power_print(f'...validating', logfile)
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        avg_loss_val = 0
        avg_count_val = 0
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, img_names, img_height, img_width = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            loss_net_val = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            avg_loss_val += loss_net_val.data
            avg_count_val += 1

            # TODO: mAP
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            

        val_losses.append((avg_loss_val/avg_count_val).cpu().numpy())
        power_print('[%d] time: %f val loss: %f' % (epoch, time.time()-start_time, avg_loss_val/avg_count_val), logfile)

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default)
        windowname = f'val_result/val_{epoch + 1}_{img_names[0]}'
        visualize_pred(windowname, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch % 10==9:
            #save last network
            print('saving net...')
            logfile.write(f'saving net: network_{epoch+1}.pth\n')
            logfile.flush()
            torch.save(network.state_dict(), f'network_{epoch+1}.pth')

    logfile.close()

else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    # logfile = open(f'test_{datetime.now().strftime("%m%d_%H_%M")}.log',  'w')
    # logfile.write(datetime.now().strftime("%Y/%m/%d %H:%M") + "\n")
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, img_names, img_height, img_width = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        # height_origin = height_[0].numpy()
        # width_origin = width_[0].numpy()
        # pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        # write_txt(pred_box_, boxs_default, pred_confidence_, img_name_[0], height_origin, width_origin,args.txt)
        
        windowname = f'test_result/test_{img_names[0]}'
        visualize_pred(windowname, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        cv2.waitKey(1000)