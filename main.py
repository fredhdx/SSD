import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from dataset import *
from model import *
from utils import *
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')  # run test evaluation
parser.add_argument('--testrun', action='store_true')  # run a short test with 30 images
parser.add_argument('--map', action='store_true')
parser.add_argument('--pth')  # checkpoint file
parser.add_argument('--testdir')  # optional testdir name: data/testname
parser.add_argument('--nmsoverlap')
args = parser.parse_args()


class_num = 4 # cat dog person background
num_epochs = 100
batch_size = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nms_overlap = 0.2
if args.nmsoverlap:
    try:
        float(args.nmsoverlap)
        nms_overlap = float(args.nmsoverlap)
    except Exception:
        pass

#Create network
boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
network = SSD(class_num)
network.to(device=device)
cudnn.benchmark = True

# create directory to store images
dirpath = os.getcwd()
if not os.path.isdir(dirpath + "/val_result"):
    os.mkdir(dirpath + "/val_result")
if not os.path.isdir(dirpath + "/train_result"):
    os.mkdir(dirpath + "/train_result")
if not os.path.isdir(dirpath + "/test_result"):
    os.mkdir(dirpath + "/test_result")

# open log file

print('--------- SETTING UP ----------')
print(f'args: {args}')
if not args.test:
    # open log file
    lfn = f'train_{datetime.now().strftime("%m%d_%H")}.log'
    lf_mode = 'a' if os.path.isfile(lfn) else 'w'
    logfile = open(lfn, lf_mode)

    # dataset
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320, testrun=args.testrun)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320, testrun=args.testrun)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # optimizer 
    optimizer = optim.Adam(network.parameters(), lr=lr)
    
    # losses 
    start_time = time.time()
    train_losses = []  # accu train loss over epochs
    val_losses = []  # accu val loss over epochs

    train_size = len(dataloader)
    start_epoch = 0

    # load from saved state
    if args.pth:
        power_print(f'loading state from {args.pth}', logfile)
        _train_losses, _val_losses, _start_epoch = load_state(network, optimizer, args.pth)
        if _start_epoch is not None:
            train_losses = _train_losses
            val_losses = _val_losses
            start_epoch = _start_epoch + 1

    power_print(datetime.now().strftime("%Y/%m/%d %H:%M"), logfile)
    for epoch in range(start_epoch, num_epochs):
        power_print(f'---------- EPOCH {epoch + 1} ------------', logfile)
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        power_print(f'...training', logfile)
        for i, data in enumerate(dataloader, 0):
            tmp_time = time.time()

            images_, ann_box_, ann_confidence_, img_names, img_height, img_width = data
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            # print(f'ann_confidence shape: {ann_confidence.shape}')
            # print(f'ann_box shape: {ann_box.shape}')

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)

            # print(f'pred_confidence shape: {pred_confidence.shape}')
            # print(f'pred_box shape: {pred_box.shape}')

            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
            # if (i+1) % 100 == 0:
            print(f'...cost {int(time.time() - tmp_time)} seconds for image {i+1}/{train_size}')

        train_losses.append((avg_loss/avg_count).cpu().numpy())
        power_print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count), logfile)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=nms_overlap)
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
            images = images_.to(device)
            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

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

        #visualize only the first image in the batch
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_, pred_box_, boxs_default, overlap=nms_overlap)
        windowname = f'val_result/val_{epoch + 1}_{img_names[0]}'
        visualize_pred(windowname, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        #save weights
        if epoch % 10==9:
            #save last network
            print('saving net...')
            logfile.write(f'saving net: network_{epoch+1}.pth\n')
            logfile.flush()
            save_state(network, optimizer, epoch, train_losses, val_losses, f'network_{epoch+1}.pth')

    logfile.close()

else:
    #TEST
    if args.map:
        dataset_test = COCO(f"data/train/images/", f"data/train/annotations/", class_num, boxs_default,
                            train = False, image_size=320)  # use validation set
    else:
        testdir = args.testdir if args.testdir else 'test'
        dataset_test = COCO(f"data/{testdir}/images/", f"", class_num, boxs_default, train = False,
                            image_size=320)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    if args.pth:
        checkpoint = torch.load(args.pth)
        if "network_state" in checkpoint:
            network.load_state_dict(checkpoint["network_state"])
        else:
            network.load_state_dict(checkpoint)
    else:
        raise Exception('checkpoint file name must be entered via "--pth /{name/}" for testing.')


    network.eval()

    total_annbox = []
    total_annconf = []
    total_predbox = []
    total_predconf = []

    for i, data in enumerate(dataloader_test, 0):
        # print(f'batch {i+1}')
        images_, ann_box_, ann_confidence_, img_names, img_height, img_width = data
        images = images_.to(device)
        ann_box = ann_box_.to(device)
        ann_confidence = ann_confidence_.to(device)
        pred_confidence, pred_box = network(images)

        total_annbox.append(ann_box.detach().cpu().numpy())
        total_annconf.append(ann_confidence.detach().cpu().numpy())
        total_predbox.append(pred_box.detach().cpu().numpy())
        total_predconf.append(pred_confidence.detach().cpu().numpy())


        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        # draw a figure for last image in batch
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        pred_confidence_, pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        windowname = f'test_result/test_{img_names[0]}'
        visualize_pred(windowname, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        # cv2.waitKey(1000)


    # get mAP
    # ann_box, ann_confidence, pred_confidence_, pred_box_: (32, 540, 4)
    if args.map:
        print('compile torch output to mAP inputs')
        pred_boxes, true_boxes = build_map_dataset(total_annbox, total_annconf, total_predbox, total_predconf,
                                                boxs_default, num_classes=class_num)

        print('calculating mAP score')
        mAP_scores = []
        _score = generate_mAP(pred_boxes, true_boxes, threshold=0.5, num_classes=4)
        for _threshold in np.linspace(0.5, 0.95, 10):
            _score = generate_mAP(pred_boxes, true_boxes, threshold=_threshold, num_classes=4)
            print(f'threshold={_threshold}, score={_score}')
            mAP_scores.append(_score)
        print(f'average mAP: {sum(mAP_scores)/len(mAP_scores)}')
