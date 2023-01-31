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
import numpy as np
import os
import cv2
import random

from PIL import Image


#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.

    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    def _getbox(x_center, y_center, width, height):
        x_min = max(x_center - width/2, 0.0)
        y_min = max(y_center - height/2, 0.0)
        x_max = min(x_center + width/2, 1.0)
        y_max = min(y_center + height/2, 1.0)
        return [x_center, y_center, width, height, x_min, y_min, x_max, y_max]
        

    cell_num = 10*10 + 5*5 + 3*3 + 1*1
    box_num = 4*cell_num

    boxes = np.zeros((box_num, 8), dtype=np.float32)

    count = 0
    for i, layer_size in enumerate(layers):
        # each layer: create 4 boxes for each cell
        lsize = large_scale[i]
        ssize = small_scale[i]
        delta = 1 / layer_size  # for simplicity, assume scale is 1 x 1
        base_c = delta / 2

        for j in range(layer_size):
            for k in range(layer_size):
                # each cell
                x_center = base_c + j * delta
                y_center = base_c + k * delta

                box1 = _getbox(x_center, y_center, ssize, ssize)
                box2 = _getbox(x_center, y_center, lsize, lsize)
                box3 = _getbox(x_center, y_center, lsize*np.sqrt(2), lsize/np.sqrt(2))
                box4 = _getbox(x_center, y_center, lsize/np.sqrt(2), lsize*np.sqrt(2))

                boxes[count, :] = box1
                boxes[count + 1, :] = box2
                boxes[count + 2, :] = box3
                boxes[count + 3, :] = box4

                count += 4
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)  # shape=(num_boxes, )
    ious_true = ious > threshold

    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    confidence = [0, 0, 0, 0]
    confidence[cat_id] = 1

    gw = x_max - x_min
    gh = y_max - y_min
    gx = x_min + gw / 2.0
    gy = y_min + gh / 2.0

    update_list = list(np.where(ious_true)[0])
    if not update_list:
        update_list = [np.argmax(ious)]
    for idx in update_list:
        # encode category
        ann_confidence[idx, :] = confidence
        # encode attributes
        px, py, pw, ph = boxs_default[idx, 0:4]
        tx = (gx - px)/pw
        ty = (gy - py)/ph
        tw = np.log(gw/pw)
        th = np.log(gh/ph)
        ann_box[idx, :] = [tx, ty, tw, th]
    
class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, 
                 image_size=320, brightness=0.24, contrast=0.15,
                 saturation=0.3, hue=0.14, testrun=False):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5  # iou threshold
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        if testrun:  # limit data size for fast code test
            self.img_names = self.img_names[:50]

        # color jittering
        self.colorJitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        total_images = len(self.img_names)
        if self.train:
            self.img_names = self.img_names[:int(0.9 * total_images)]
        else:
            self.img_names = self.img_names[int(0.9 * total_images):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # relative attributes: [tx, ty, tw, th]
        # one-hot vector for defualt bbox
        ann_box = np.zeros([self.box_num, 4], np.float32) 
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)
        #   [1,0,0,0] -> cat
        #   [0,1,0,0] -> dog
        #   [0,0,1,0] -> person
        #   [0,0,0,1] -> background

        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3]+"txt"
        
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #    get gt scaled bbox
        #    to use function "match":
        # match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        # note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        # for each gt bounding box
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.

        img = cv2.imread(img_name)  # (h, w, c), dtype=uint8, BGR
        img_height, img_width, img_c = img.shape

        class_id, x_min, y_min, x_max, y_max = [], [], [], [], []

        # read annotations
        with open(ann_name, 'r') as f:
            for line in f:
                _gt_tuple = line.strip().split(" ")
                if len(_gt_tuple) != 5:  # skip if invalid
                    continue

                # TODO: confirm entry identify
                class_id.append(int(_gt_tuple[0]))
                gx_min, gy_min, gw, gh = [float(_) for _ in _gt_tuple[1:]]

                x_min.append(gx_min)
                y_min.append(gy_min)
                x_max.append(gx_min + gw)
                y_max.append(gy_min + gh)

        class_id = np.asarray(class_id)
        x_min = np.asarray(x_min)
        y_min = np.asarray(y_min)
        x_max = np.asarray(x_max)
        y_max = np.asarray(y_max)
        
        # random crop (training + validation)
        if self.train:
            rand_x_min = np.random.randint(0, np.min(x_min))
            rand_y_min = np.random.randint(0, np.min(y_min))
            rand_x_max = np.random.randint(np.max(x_max), img_width)
            rand_y_max = np.random.randint(np.max(y_max), img_height)

            # crop image
            img_width = rand_x_max - rand_x_min
            img_height = rand_y_max - rand_y_min
            img = img[rand_y_min:rand_y_max, rand_x_min:rand_x_max, :]

            # crop label
            x_min -= rand_x_min
            y_min -= rand_y_min
            x_max -= rand_x_min
            y_max -= rand_y_min

            # TODO: additional affine transfomration

        # scale annotation to (0, 1)
        x_min = x_min / img_width
        y_min = y_min / img_height
        x_max = x_max / img_width
        y_max = y_max / img_height

        for i in range(len(x_max)):
            match(ann_box, ann_confidence, self.boxs_default, self.threshold,
                  class_id[i], x_min[i], y_min[i], x_max[i], y_max[i])

        # final processisng
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size))
        ])

        image = preprocessing(img)
        image = image / 255.

        # if self.train:
        #     image = self.colorJitter(image)

        return image, torch.from_numpy(ann_box), torch.from_numpy(ann_confidence), self.img_names[index][:-3], img_height, img_width
