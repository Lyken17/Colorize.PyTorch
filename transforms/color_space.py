import math

import numpy as np

import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F


class RGB2GRAY(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        import cv2, numpy as np

        npimg = tensor.numpy()
        npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
        npimg = cv2.cvtColor(npimg, cv2.COLOR_GRAY2RGB)
        tensor = torch.from_numpy(npimg)
        return tensor


class Gamma(object):
    def __init__(self, power=2):
        self.power = power

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        return tensor ** self.power


class Linearize(object):
    def __init__(self, power=2):
        self.power = power

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = 0.055

        tensor[tensor < 0.04045] /= 12.92
        tensor[tensor > 0.04045] = ((tensor[tensor > 0.04045] + alpha) / (1 + alpha)) ** 2.4

        return tensor


class __color_space_convert(object):
    def __init__(self, mat=None):
        self.mat = mat

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        if self.mat is None:
            raise NotImplementedError
        # tensor: 3xHxW
        s = tensor.size()

        t = tensor.permute(1, 2, 0).view(-1, s[0])
        res = torch.mm(t, self.mat).view(s[1], s[2], 3).permute(2, 0, 1)

        return res


class SRGB2XYZ(__color_space_convert):
    # D65
    # https://en.wikipedia.org/wiki/SRGB
    def __init__(self):
        mat = torch.Tensor(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]]
        )
        super(SRGB2XYZ, self).__init__(mat=mat)


class XYZ2CIE(__color_space_convert):
    def __init__(self):
        mat = torch.Tensor(
            [[0.4002, 0.7076, -0.0808],
             [-0.2263, 1.1653, 0.0457],
             [0.0, 0.0, 0.9182]]
        )
        super(XYZ2CIE, self).__init__(mat=mat)
