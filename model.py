import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

    bn, num_boxes, num_classes = pred_confidence.shape

    # reshape inputs
    pred_confidence = pred_confidence.reshape((-1, num_classes))
    pred_box = pred_box.reshape((-1, 4))
    ann_confidence = ann_confidence.reshape((-1, num_classes))
    ann_box = ann_box.reshape((-1, 4))

    # get non-empty indices
    indicies_obj = torch.where(ann_confidence[:, -1] == 0)[0]

    # cls loss
    loss_cls_obj = F.cross_entropy(pred_confidence[indicies_obj], ann_confidence[indicies_obj])
    loss_cls_noobj = F.cross_entropy(pred_confidence[~indicies_obj], ann_confidence[~indicies_obj])
    loss_cls = loss_cls_obj + 3 * loss_cls_noobj

    # box loss
    loss_box = F.smooth_l1_loss(pred_box[indicies_obj], ann_box[indicies_obj])

    return loss_cls + loss_box



class Conv2dBnRelu(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=cin, out_channels=cout,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class Conv2dReshape(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.conv_box = nn.Conv2d(256, 16, kernel_size, stride, padding)
        self.conv_conf = nn.Conv2d(256, 16, kernel_size, stride, padding)

    def forward(self, x):
        box = self.conv_box(x)
        conf = self.conv_conf(x)
        bn, c, w, h = box.shape
        box = box.reshape((bn, c, w*h))
        conf = conf.reshape((bn, c, w*h))
        return box, conf 

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        # sequence 1
        self.conv_base = nn.Sequential(
            # 3, 320 -> 64, 160
            Conv2dBnRelu(3, 64, 3, 2, 1), # cin, cout, kernel, stride, padding
            Conv2dBnRelu(64, 64, 3, 1, 1),
            Conv2dBnRelu(64, 64, 3, 1, 1),

            # 64, 160 -> 128, 80
            Conv2dBnRelu(64, 128, 3, 2, 1),
            Conv2dBnRelu(128, 128, 3, 1, 1),
            Conv2dBnRelu(128, 128, 3, 1, 1),

            # 128, 80 -> 256, 40
            Conv2dBnRelu(128, 256, 3, 2, 1),
            Conv2dBnRelu(256, 256, 3, 1, 1),
            Conv2dBnRelu(256, 256, 3, 1, 1),

            # 256, 40 -> 512, 20
            Conv2dBnRelu(256, 512, 3, 2, 1),
            Conv2dBnRelu(512, 512, 3, 1, 1),
            Conv2dBnRelu(512, 512, 3, 1, 1),

            # 512, 20 -> 256, 10
            Conv2dBnRelu(512, 256, 3, 2, 1)
        )

        # sequence 2
        # 256, 10, 10 -> 256, 10, 10
        self.conv21 = Conv2dBnRelu(256, 256, 1, 1, 0)
        # 256, 10, 10 -> 256, 5, 5
        self.conv22 = Conv2dBnRelu(256, 256, 3, 2, 1)

        # sequence 3
        # 256, 5, 5 -> 256, 5, 5
        self.conv31 = Conv2dBnRelu(256, 256, 1, 1, 0)
        # 256, 5, 5 -> 256, 3, 3
        self.conv32 = Conv2dBnRelu(256, 256, 3, 1, 0)

        # sequence 4
        # 256, 3, 3 -> 256, 3, 3
        self.conv41 = Conv2dBnRelu(256, 256, 1, 1, 0)
        # 256, 3, 3 -> 256, 1, 1
        self.conv42 = Conv2dBnRelu(256, 256, 3, 1, 0)

        # divergence branches
        # input: kernel size, stride, padding
        self.divergence10 = Conv2dReshape(3, 1, 1)  # 10x10, (N, 16, 100)
        self.divergence5 = Conv2dReshape(3, 1, 1)  # 5x5, (N, 16, 25)
        self.divergence3 = Conv2dReshape(3, 1, 1)  # 3x3, (N, 16, 9)
        self.divergence1 = Conv2dReshape(1, 1, 0)  # 1x1, (N, 16, 1)


    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]

        # base conv (sequence 1)
        x = self.conv_base(x)

        # branch: 10x10
        box10, conf10 = self.divergence10(x)

        # sequence 2
        x = self.conv21(x)
        x = self.conv22(x)
        # branch: 5x5
        box5, conf5 = self.divergence5(x)

        # sequence 3
        x = self.conv31(x)
        x = self.conv32(x)
        # branch: 3x3
        box3, conf3 = self.divergence3(x)

        # sequence 4
        x = self.conv41(x)
        x = self.conv42(x)
        # branch: 1x1
        box1, conf1 = self.divergence1(x)

        # concatenate
        bboxes = torch.cat([box10, box5, box3, box1], dim=2)
        confidence = torch.cat([conf10, conf5, conf3, conf1], dim=2)

        # permute
        bboxes = torch.permute(bboxes, [0, 2, 1])
        confidence = torch.permute(confidence, [0, 2, 1])

        # reshape
        bn, _, _ = bboxes.shape
        bboxes = bboxes.reshape((bn, -1, 4))
        confidence = confidence.reshape((bn, -1, self.class_num))

        # softmax
        confidence = F.softmax(confidence, dim=2)
        
        return confidence, bboxes



if __name__ == "__main__":
    network = SSD(4)
    img = torch.randn((8, 3, 320, 320))
    confidence, bboxes = network(img)






