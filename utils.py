import numpy as np
import cv2
from dataset import iou
from collections import Counter
import torch
from matplotlib import pyplot as plt


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def calculate_iou(box1, x_min, y_min, x_max, y_max):
    inter = np.maximum(np.minimum(box1[2],x_max)-np.maximum(box1[0],x_min),0)\
            * np.maximum(np.minimum(box1[3],y_max)-np.maximum(box1[1],y_min),0)
    area_a = (box1[2]-box1[0])*(box1[3]-box1[1])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    image_ = image_ * 255
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num - 1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8) # (h, w, c)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]  # gt
    image2[:]=image[:]  # gt default boxes
    image3[:]=image[:]  # predicated bbox
    image4[:]=image[:]  # predicated default boxes
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

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.5):
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
    confidence  = np.zeros(confidence_.shape, dtype=np.float32)
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
            overlap_idx = np.where(ious > overlap)[0]
            box_[overlap_idx, :] = [0.] * 4
            confidence_[overlap_idx, :] = [0, 0, 0, 1]
            gt_boxes[overlap_idx, :] = [0.] * 8
        else:
            break

    return confidence, bbox

def build_map_dataset(total_annbox, total_annconf, total_predbox, total_predconf, boxs_default, num_classes=4):
    ''' input: list of batched modle output, [(32, 540, 4)]
        We want to loop through every batch, every image then extract gt boxes and predictions for each image.
        output: two lists of gt boxes and predicted boxes
    '''
    pred_boxes = []
    true_boxes = []
    unique_class = set()

    # generate true boxes
    image_id = 0

    print(f"there are {len(total_annbox)} batches")

    # for each batch 
    for batch_idx in range(len(total_annbox)):
        batch_annconf = total_annconf[batch_idx]  # (32, 540, 4)
        batch_predconf = total_predconf[batch_idx]
        batch_annbox = total_annbox[batch_idx]
        batch_predbox = total_predbox[batch_idx]

        # for each image in batch
        for image_idx in range(batch_annbox.shape[0]):

            # one image: (540, 4)
            annbox = batch_annbox[image_idx, :, :].reshape((-1, 4))
            annconf = batch_annconf[image_idx, :, :].reshape((-1, 4))
            predbox = batch_predbox[image_idx, :, :].reshape((-1, num_classes))
            predconf = batch_predconf[image_idx, :, :].reshape((-1, num_classes))

            predconf, predbox = non_maximum_suppression(predconf, predbox, boxs_default)

            # convert to absolute coordinate
            annbox = relative_boxes_to_absolute(annbox, boxs_default)[:, 4:]   # (540, 4)
            predbox = relative_boxes_to_absolute(predbox, boxs_default)[:, 4:]

            # restore gt boxes
            indicies_gt = np.where(annconf[:, -1] == 0)[0]
            for gt_idx in indicies_gt:
                class_id = np.argmax(annconf[gt_idx])
                true_boxes.append([image_id, class_id, 1.0, 
                                   float(annbox[gt_idx, 0]), 
                                   float(annbox[gt_idx, 1]), 
                                   float(annbox[gt_idx, 2]), 
                                   float(annbox[gt_idx, 3])])

            # convert predictions
            indices_pred = np.argwhere(predconf > 0.5)
            for obj_idx in indices_pred:
                _idx = obj_idx[0]
                class_id = obj_idx[1]
                if class_id == num_classes - 1:  # skip background
                    continue
                pred_score = predconf[_idx, class_id]
                pred_boxes.append([image_id, class_id, pred_score, 
                                   float(predbox[_idx, 0]), 
                                   float(predbox[_idx, 1]), 
                                   float(predbox[_idx, 2]), 
                                   float(predbox[_idx, 3])])
                
                unique_class.add(class_id)
            
            image_id += 1  # increment image id
        
    print(f"unique predicted classes: {unique_class}")

    return pred_boxes, true_boxes

def generate_mAP(pred_boxes, true_boxes, threshold=0.5, num_classes=4):
    # pred_boxes: [[image_id, class_id, pred_score, x_min, y_min, x_max, y_max]]
    # true_boxes: [[image_id, class_idx, 1, x_min, y_min, x_max, y_max]]
    average_precisions = []
    colormap = ['red', 'green', 'blue']
    fig = plt.figure()
    plt.title("Precision Recall Curve\nThreshold={threshold}")
    epsilon = 1e-6

    print(f"size of pred_boxes: {len(pred_boxes)}")
    print(f"size of true_boxes: {len(true_boxes)}")


    # for each class, we look at all predications that predict this class and gt that belong to this class
    #   pred for image 1, pred for image 2, gt for image 5, gt for img 2, etc..
    for c in range(num_classes - 1):  # [0, 1, 2, 3]
        # filter predication and ground truth by class c
        predications = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]
        
        # set up image filtering counter: what image has how many gt truths
        # [0, 0, ..] if each gt has been assigned 
        # { 
        #  image 1: [0, 0, 0],          # number of gt boxes
        #  image 2: [0,0,0,0,0]
        # }  
        track_gt_per_image = Counter(gt[0] for gt in ground_truths)
        for key, val in track_gt_per_image.items():
            track_gt_per_image[key] = torch.zeros(val)

        # sort predication by confidence score (for all images)
        predications.sort(key=lambda x: x[2], reverse=True)

        # get TP and FP for (all images)
        # TP: this predication is correct (find a matching gt box from the same image)
        # FP: this predication is not correct (no matching gt box found)
        TP = torch.zeros((len(predications)))  # image 1 has 5 predictions, [0, 0, 0, 0, 0, 0]
        FP = torch.zeros((len(predications)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        # For each predication (from all images)
        for pred_idx, pred in enumerate(predications):
            # this prediction's components
            image_id = pred[0]

            # find matching gt for this image 
            target_gts = [box for box in ground_truths if box[0] == image_id]
            best_iou = 0

            # find the most likely gt box for this prediction on the same image
            for idx, gt in enumerate(target_gts):
                # iou between prediction and grouth truth absolute positions [x1, y1, x2, y2]
                iou = calculate_iou(np.array(pred[3:]), gt[3], gt[4], gt[5], gt[6])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx # out of target_gts

            # check if the predication can be deemed correct w.r.t. the most likely gt box
            if best_iou > threshold:
                # amount_bboxes for image 1, test if this gt bbox has been used.
                gts_in_image = track_gt_per_image[image_id]
                if gts_in_image[best_gt_idx] == 0:
                    TP[pred_idx] = 1  # so this predication is a true positive.
                    gts_in_image[best_gt_idx] = 1
                else: # this gt box has been assigned as correct prediction to another prediction
                    FP[pred_idx] = 1 
            else:
                FP[pred_idx] = 1
            
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))
        plt.plot(recalls, precisions, color=colormap[c], label=f"Class: {c}")
    
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"Precision-Recall_{threshold}.jpg")
    return sum(average_precisions) / len(average_precisions)


def save_state(network, optimizer, epoch, train_losses, val_losses, fn):
    torch.save({
        'epoch': epoch,
        'network_state': network.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, fn)


def load_state(network, optimizer, fn):
    ckp = torch.load(fn)
    if "optimizer_state" in ckp:
        network.load_state_dict(ckp["network_state"])
        optimizer.load_state_dict(ckp["optimizer_state"])
        return ckp["train_losses"], ckp["val_losses"], int(ckp["epoch"])
    else:
        network.load_state_dict(ckp)
        return None, None, None  # fallback to old state file


# print to log file as well
def power_print(msg, fp):
    print(msg)
    fp.write(msg + "\n")
    fp.flush()