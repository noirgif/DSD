import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class DSD_Dropout(nn.module):
"""
use phase_change to change the phase in the training
"""
    (DENSE, SPARSE) = (0, 1)
    phase = DENSE

    def __init__(self, drop_ratio)
        super(DSD_Dropout, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if self.phase is DSD_Dropout.DENSE:
            return x
        else:
# calculate the k using x's size(note that x is unidimensional)
            k = drop_ratio * x.size()[0]
# get the kth smallest element
            kth_value = torch.abs(x).kthvalue(k)[0]
            mask = nn.Threshold(kth_value, 0)
"""
    filter those smaller than kth_value
    and larger than -kth_value
"""
            x = mask(x) + mask(-x)
            return x
            
 
class AlexNet(nn.module):
"""
    Extract the feature part as a module
    rewrite the classifier part
"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

# substitute the dropout with DSD layer
        self.classifier = nn.Sequential(
            DSD_Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            DSD_Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

