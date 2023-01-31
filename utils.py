import numpy as np
import cv2
from dataset import iou
from collections import Counter
import torch


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8) # (h, w, c)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    ann_box = relative_boxes_to_absolute(ann_box, boxs_default)
    pred_box = relative_boxes_to_absolute(pred_box, boxs_default)
    height, width, _  = image.shape
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j] > 0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)

                color = colors[j] 
                #image1: draw ground truth bounding boxes on image1
                (xmin, ymin) = (int(ann_box[i, 4] * width), int(ann_box[i, 5] * height))
                (xmax, ymax) = (int(ann_box[i, 6] * width), int(ann_box[i, 7] * height))
                image1 = cv2.rectangle(image1, (xmin, ymin), (xmax, ymax), color, thickness=2)
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                (xmin, ymin) = (int(boxs_default[i, 4] * width), int(boxs_default[i, 5] * height))
                (xmax, ymax) = (int(boxs_default[i, 6] * width), int(boxs_default[i, 7] * height))
                image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), color, thickness=2)

    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                color = colors[j] 
                #image3: draw network-predicted bounding boxes on image3
                (xmin, ymin) = (int(pred_box[i, 4] * width), int(pred_box[i, 5] * height))
                (xmax, ymax) = (int(pred_box[i, 6] * width), int(pred_box[i, 7] * height))
                image3 = cv2.rectangle(image3, (xmin, ymin), (xmax, ymax), color, thickness=2)
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                (xmin, ymin) = (int(boxs_default[i, 4] * width), int(boxs_default[i, 5] * height))
                (xmax, ymax) = (int(boxs_default[i, 6] * width), int(boxs_default[i, 7] * height))
                image4 = cv2.rectangle(image4, (xmin, ymin), (xmax, ymax), color, thickness=2)
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    cv2.imwrite(windowname + ".jpg", image)

def relative_boxes_to_absolute(box_, boxs_default):
    dx, dy, dw, dh = box_[:, 0], box_[:, 1], box_[:, 2], box_[:, 3]
    px, py, pw, ph = boxs_default[:, 0], boxs_default[:, 1], boxs_default[:, 2], boxs_default[:, 3]

    gx_hat = pw * dx + px
    gy_hat = ph * dy + py
    gw_hat = pw * np.exp(dw)
    gh_hat = ph * np.exp(dh)

    gt_boxes = np.zeros(boxs_default.shape, dtype=np.float32)
    gt_boxes[:, 0] = gx_hat
    gt_boxes[:, 1] = gy_hat
    gt_boxes[:, 2] = gw_hat
    gt_boxes[:, 3] = gh_hat
    gt_boxes[:, 4] = gx_hat - gw_hat/2
    gt_boxes[:, 5] = gy_hat - gh_hat/2
    gt_boxes[:, 6] = gx_hat + gw_hat/2
    gt_boxes[:, 7] = gy_hat + gh_hat/2

    return gt_boxes

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]

    #boxs_default -- default bounding boxes, [num_of_boxes, 8]

    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    # recover gt boxes from prediction: gt_boxes
    gt_boxes = relative_boxes_to_absolute(box_, boxs_default)

    # output
    bbox = np.zeros(box_.shape, dtype=np.float32)
    confidence  = np.zeros(box_.shape, dtype=np.float32)
    confidence[:, -1] = 1

    # non-maximum_suppresion algorithm
    while True:
        # find flattened index of maximum probability excluding background class
        highest = np.argmax(confidence_[:, 0:-1])
        idx, cls = divmod(highest, confidence_[:, 0:-1].shape[1])

        # carrying a bbox
        if confidence_[idx, cls] >= threshold:
            # copy over to output
            bbox[idx, :] = box_[idx, :]
            confidence[idx, :] = confidence_[idx, :]

            # throw out boxes with iou overlap with found high prob box
            # (including itself)
            ious = iou(gt_boxes, gt_boxes[idx, 4], gt_boxes[idx, 5], gt_boxes[idx, 6],
                       gt_boxes[idx, 7]) # iou(boxs_default, x_min, y_min, x_max, y_max)
            ious = np.where(ious > overlap)[0]
            box_[ious, :] = [0.] * 4
            confidence_[ious, :] = [0, 0, 0, 1]
            gt_boxes[ious, :] = [0.] * 8
        else:
            break

    return confidence, bbox


def generate_mAP(pred_boxes, true_boxes, threshold=0.5, num_classes=4):
    #TODO: Generate mAP
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        grount_truths =[] 

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                grount_truths.append(true_box)

        
        amount_bboxes = Counter(gt[0] for gt in grount_truths)

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zero(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(grount_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in grount_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]))

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            
            if best_iou > threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))

            average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)
                









