from numpy import frombuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def argmax(net_output): 
    shp_x = net_output.shape
    output_onehot = torch.zeros(shp_x)
    
    if net_output.device.type == "cuda":
        output_onehot = output_onehot.cuda(net_output.device.index)    

    output_onehot.scatter_(1, net_output.argmax(dim=1, keepdim=True).long(), 1)
    return output_onehot


def sum_tensor(inp, axes, keepdim=False):
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, do_bg=True, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        """
        super(GDiceLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, gt, net_output):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        input = flatten(net_output)
        target = flatten(gt)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)
        
        intersect = (input * target).sum(-1) * class_weights
        denominator = (input + target).sum(-1) * class_weights

        if not self.do_bg: 
            intersect = intersect[1:]
            denominator = intersect[1:]
                
        intersect = intersect.sum()
        denominator = denominator.sum()
        return  - 2. * intersect / denominator.clamp(min=self.smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False):
        super(SoftDiceLoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, y, x, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        return 1 - dc.mean()


class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1., square=False):
        super(IoULoss, self).__init__()
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, y, x, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)


        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        return 1 - iou.mean()


class Accuracy: 
    def __init__(self, apply_nonlin=None): 
        self.apply_nonlin = apply_nonlin

    def __call__(self, y, x): 
        if self.apply_nonlin is not None: 
            x = self.apply_nonlin(x)
            
        x = x.argmax(dim=1, keepdim=False)
        y = y.argmax(dim=1, keepdim=False)

        return (x == y).sum()/ x.numel()


class BceLoss(nn.Module): 
    def __init__(self): 
        super(BceLoss, self).__init__()

    def forward(self, y, x):
        return F.nll_loss(F.log_softmax(x, dim=1), y.argmax(dim=1))


class BceDiceLoss(nn.Module): 
    def __init__(self): 
        super(BceDiceLoss, self).__init__()
        self.diceloss = SoftDiceLoss(nn.Softmax(dim=1), batch_dice=True, do_bg=False, smooth=1., square=True)
        self.bceloss = BceLoss()

    def forward(self, y, x): 
        return self.diceloss(y, x) + self.bceloss(y, x)


class BceIouLoss(nn.Module): 
    def __init__(self): 
        super(BceIouLoss, self).__init__()
        self.iouloss = IoULoss(nn.Softmax(dim=1), batch_dice=True, do_bg=False, smooth=1e-6, square=True)
        self.bceloss = BceLoss()

    def forward(self, y, x): 
        return self.iouloss(y, x) + self.bceloss(y, x)
    

def focal_loss(targets, predictions, alpha=1, gamma=2, epsilon=1e-6, softmax=True):
    if softmax:
        ce = -F.nll_loss(F.log_softmax(predictions, dim=1), targets.argmax(dim=1), reduction='none')
    else:
        ce = -F.nll_loss(torch.log(predictions), targets.argmax(dim=1), reduction='none')
    return torch.mean(-alpha * ce * torch.pow(1 - torch.exp(ce), gamma))
    