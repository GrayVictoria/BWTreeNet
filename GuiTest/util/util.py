# coding:utf-8
import numpy as np
# import chainer
from PIL import Image
import torch
# from ipdb import set_trace as st


def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(num_classes, logits, labels):
    logits = logits.argmax(0)
    # print (logits.shape)
    # print(logits)
    # print (labels.shape)
    intersect = logits[logits == labels]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        logits.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        labels.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label 

def total_intersect_and_union(num_classes,logits,labels):
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    


def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc

def calculate_mean_iou(logits,labels):
    return miou


def calculate_mean_accuracy(logits, labels):
    return macc


def calculate_class_accuracy(logits, labels):
    return class_acc


def calculate_class_iou(logits, labels):
    return class_iou


# def calculate_result(cf):
#     n_class = cf.shape[0]
#     conf = np.zeros((n_class,n_class))
#     IoU = np.zeros(n_class)
#     conf[:,0] = cf[:,0]/cf[:,0].sum()
#     for cid in range(1,n_class):
#         if cf[:,cid].sum() > 0:
#             conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
#             IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
#     overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
#     acc = np.diag(conf)

#     return overall_acc, acc, IoU

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    IoU = np.zeros(n_class)
    conf[:,0] = cf[:,0]/cf[:,0].sum()
    for cid in range(0,n_class):
        if cf[:,cid].sum() > 0:
            conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
            IoU[cid]  = cf[cid,cid]/(cf[cid,0:].sum()+cf[0:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[0:,0:]).sum()/cf[0:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU



# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(names, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(names[i].replace('.png', '_pred.png'))


#直方图均衡化
class EqualizeHist:
    def __init__(self, image, bins=65536, normalize_max=255, normalize_type='uint8'):
        self.image = image
        self.bins = image.max()+1
        self.normalize_max = normalize_max
        self.normalize_type = normalize_type

    def get_histogram(self, image):
        # array with size of bins, set to zeros
        histogram = np.zeros(self.bins)
        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1
        # return our final result
        return histogram

    def cumsum(self, a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def operation(self):
        flat = self.image.flatten()
        hist = self.get_histogram(flat)
        # execute the fn
        cs = self.cumsum(hist)
        # numerator & denomenator
        nj = (cs - cs.min()) * self.normalize_max
        N = cs.max() - cs.min()
        # re-normalize the cdf
        cs = nj / N
        cs = cs.astype(self.normalize_type)
        image_new = cs[flat]
        image_new = np.reshape(image_new, self.image.shape)
        return image_new

#百分比截断


class TruncatedLinearStretch:
    def __init__(self, image, truncated_value=2, max_out=255, min_out=0, normalize_type='uint8'):
        self.image = image
        self.truncated_value = truncated_value
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        truncated_down = np.percentile(self.image, self.truncated_value)
        truncated_up = np.percentile(self.image, 100 - self.truncated_value)
        image_new = (self.image - truncated_down) / (truncated_up -
                                                     truncated_down) * (self.max_out - self.min_out) + self.min_out
        image_new[image_new < self.min_out] = self.min_out
        image_new[image_new > self.max_out] = self.max_out
        image_new = image_new.astype(self.normalize_type)
        return image_new

##标准差拉伸


class StandardDeviation:
    def __init__(self, image, parameter=2, max_out=255, min_out=0, normalize_type='uint8'):
        self.image = image
        self.parameter = parameter
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        Mean = np.mean(self.image)
        StdDev = np.std(self.image, ddof=1)
        ucMax = Mean + self.parameter * StdDev
        ucMin = Mean - self.parameter * StdDev
        k = (self.max_out - self.min_out) / (ucMax - ucMin)
        b = (ucMax * self.min_out - ucMin * self.max_out) / (ucMax - ucMin)
        if (ucMin <= 0):
            ucMin = 0

        image_new = np.select([self.image == self.min_out, self.image <= ucMin, self.image >= ucMax,  k*self.image+b < self.min_out, k*self.image+b > self.max_out,
                               (k*self.image+b > self.min_out) & (k*self.image+b < self.max_out)],
                              [self.min_out, self.min_out, self.max_out, self.min_out, self.max_out, k * self.image + b], self.image)
        image_new = image_new.astype(self.normalize_type)
        return image_new
