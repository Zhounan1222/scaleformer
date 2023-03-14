import torch
import logging
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import nn


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def DiceLoss(output, target, smooth=1e-5):
    target = target.float()
    intersection = torch.sum(output * target)
    dice = (2 * intersection + smooth) / (output.pow(2).sum() + target.pow(2).sum() + smooth)
    return 1.0 - dice


class Criterion(object):
    def __init__(self, smooth=1e-5, thresh=0.5):
        super(Criterion, self).__init__()
        self.smooth = smooth
        self.thresh = thresh
    
    def focal_loss(self, output, target):
        pass
    
    def sigmoid_dice(self, output, target):
        '''
        The dice loss for using softmax activation function
        :param output: (b, num_class, d, h, w)
        :param target: (b, d, h, w)
        :return: softmax dice loss
        '''
        assert output.shape == target.shape and len(target.shape) == 5
        target = target.float()
        intersection = (output * target).sum(dim=[2, 3, 4])
        num = (output.pow(2) + target.pow(2)).sum(dim=[2, 3, 4])
        dice = torch.mean((2 * intersection + self.smooth) / (num + self.smooth))
        return (1 - dice) ** 3

    def dice_metrix(self, output, target):
        assert output.shape == target.shape and len(target.shape) == 5
        output = (output > self.thresh).float()
        target = target.float()
        b = output.size(0)
        dice_1, dice_2, dice_3 = 0., 0., 0. 
        for i in range(b):
            dice_1 += self._binary_dice(output[i, 0, ...], target[i, 0, ...])
            dice_2 += self._binary_dice(output[i, 1, ...], target[i, 1, ...])
            dice_3 += self._binary_dice(output[i, 2, ...], target[i, 2, ...])
        return dice_1 / b, dice_2 / b, dice_3 / b

    def _binary_dice(self, output, target):
        if target.sum() == 0:
            if output.sum() == 0:
                return 1 
            else:
                return 0

        intersection = torch.sum(output * target)
        dice = (2 * intersection) / (output.sum() + target.sum())
        return dice

def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3


def Dual_focal_loss(output, target):
    # loss1 = Dice(output[:, 1, ...], (target == 1).float())
    # loss2 = Dice(output[:, 2, ...], (target == 2).float())
    # loss3 = Dice(output[:, 3, ...], (target == 4).float())
    
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    target = target.view(4, -1)
    output = output.view(4, -1)
    log = 1-(target - output)**2

    return -(F.log_softmax((1-(target - output)**2), 0)).mean()  # , 1-loss1.data, 1-loss2.data, 1-loss3.data


class SoftDiceLoss(nn.Module):

    def __init__(self, ignore_index=None, smooth=1.0, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            os = os.view(batch_size, -1)
            ts = target[:, i, ...].clone()
            ts = ts.view(batch_size, -1).float()

            inter = (os * ts).sum()
            union = os.sum() + ts.sum()

            loss += 1 - (2 * inter + self.smooth) / (union + self.smooth)
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class FocalLoss(nn.Module):
    epsilon = 1e-8

    def __init__(self, gamma=2, alpha=None, ignore_index=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        # assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            os = os.view(batch_size, -1).clamp(min=self.epsilon, max=1 - self.epsilon)
            ts = (target == i).float().clone()
            ts = ts.view(batch_size, -1).float()

            logpt_pos = ts * torch.log(os)
            pt_pos = torch.exp(logpt_pos)

            if self.alpha:
                logpt_pos *= self.alpha

            val = - ((1 - pt_pos) ** self.gamma) * logpt_pos

            # logpt_neg = (1 - ts) * torch.log(1 - os)
            # pt_neg = torch.exp(logpt_neg)
            #
            # if self.alpha:
            #     logpt_neg *= self.alpha
            #
            # val += - ((1 - pt_neg) ** self.gamma) * logpt_neg

            loss += val.mean()
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class ActiveContourLoss(nn.Module):

    def __init__(self, weight=1, epsilon=1e-8, ignore_index=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        # assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            ts = (target == i).float().clone()
            # ts = target[:, i, ...].clone()

            os[os >= 0.5] = 1
            os[os < 0.5] = 0

            # length term
            delta_r = os[:, 1:, :] - os[:, :-1, :]  # horizontal gradient (B, H-1, W)
            delta_c = os[:, :, 1:] - os[:, :, :-1]  # vertical gradient (B, H, W-1)

            delta_r = delta_r[:, 1:, :-2] ** 2  # (B, H-2, W-2)
            delta_c = delta_c[:, :-2, 1:] ** 2  # (B, H-2, W-2)

            delta_pred = torch.abs(delta_r + delta_c)
            length = torch.mean(torch.sqrt(delta_pred + self.epsilon))
            c_in = torch.ones_like(os)
            c_out = torch.zeros_like(os)

            region_in = torch.mean(os * (ts - c_in) ** 2)
            region_out = torch.mean((1 - os) * (ts - c_out) ** 2)
            region = region_in + region_out

            loss += self.weight * length + region
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class OneHotEncoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])

        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).float()

        return out


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices