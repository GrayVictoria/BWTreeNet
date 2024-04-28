import torch.nn.functional as F
from torch import nn
import torch

class CrossEntropy():
    def __init__(self, logits, labels):
        super(CrossEntropy, self).__init__()
        self.logits = logits
        self.labels =  labels
    def loss_function(self):
        return F.cross_entropy(self.logits,self.labels)


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        # source = source.argmax(1).reshape(batch_size, -1)
        # target = target.argmax(1).reshape(batch_size, -1)

        source = source.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)

        if((source != target).sum() == 0):
            return torch.tensor(0.)

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class IoULoss(nn.Module):
    # def __init__(self, weight=None, size_average=True):
    #     super(IoULoss, self).__init__()
    
    def __init__(self, logits, labels, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.logits = logits
        self.labels = labels
        self.n_classes = 2
    
    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to("cuda:0").scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def loss_function(self):
        # logit => N x Classes x H x W
        # target => N x H x W

        target = self.labels
        input = self.logits

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        IOU = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - IOU.mean()
