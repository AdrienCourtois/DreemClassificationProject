'''
Implementation of our best Wavelet model.

It reached an accuracy of 67.857% with a F1 score of
71.58% on a balanced version of the dataset.
The dataset chosen was not splited, as each person had
7*40 signals as features.
The model has around 50k parameters.
'''

from ..functional.Mish import Mish
from Refactor import Refactor
from LowPassFilter import LowPassFilter
from HighPassFilter import HighPassFilter


class EarlyConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_filters, 7*out_filters, (7, 3), stride=(7, 1)),
            nn.ReLU(),#Mish(),
            Refactor(),
            nn.BatchNorm2d(out_filters),
            nn.Conv2d(out_filters, 7*out_filters, (7, 3), stride=(7, 1)),
            Mish(),
            Refactor(),
            nn.BatchNorm2d(out_filters),
            nn.MaxPool2d((1, 2))
        )
    
    def forward(self, x):
        return self.model(x)


class LateConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, (3,3), stride=(1,1)),
            Mish(),
            nn.BatchNorm2d(out_filters),
            nn.Conv2d(out_filters, out_filters, (3,3), stride=(1,1)),
            Mish(),
            nn.BatchNorm2d(out_filters),
            nn.Conv2d(out_filters, out_filters, (3, 3), stride=(1,1)),
            Mish(),
            nn.BatchNorm2d(out_filters),
            nn.MaxPool2d((2, 2))
        )
    
    def forward(self, x):
        y = self.model(x)
        x_resize = F.interpolate(x, size=(y.size(2), y.size(3)), mode="bilinear", align_corners=False)

        return torch.cat((y, x_resize), 1)


class Model(nn.Module):
    def __init__(self, name="db3"):
        super().__init__()

        self.highpass = HighPassFilter(name)
        self.lowpass = LowPassFilter(name)

        self.low1 = EarlyConvBlock(1, 2)
        self.high1 = EarlyConvBlock(1, 2)

        self.low2 = EarlyConvBlock(2, 4)
        self.high2 = EarlyConvBlock(4, 8)

        #self.low3 = LateConvBlock(4, 8) # Real output 12
        #self.high3 = LateConvBlock(16, 32) # Real output 48 hence 60

        #self.low4 = LateConvBlock(12, 24)
        #self.high4 = LateConvBlock(60, 120)

        self.f_comp = nn.Sequential(
            nn.Conv2d(12, 20, (3,3), stride=(1,1)), #216, 12
            Mish(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d((2,2))
        )

        self.f_comp2 = nn.Sequential(
            nn.Conv2d(20, 30, (3,3), stride=(1,1)),
            Mish(),
            nn.BatchNorm2d(30),
            nn.MaxPool2d((2,2))
        )

        self.f_comp3 = nn.Sequential(
            nn.Conv2d(30, 40, (3,3), stride=(1,1)),
            Mish(),
            nn.BatchNorm2d(40),
            nn.MaxPool2d((2,2))
        )


        self.dropout = nn.Dropout(p=0.5)
        self.classif = nn.Linear(17160, 1) #216*31*11, 17160
    
    def warper(self, x, size):
        warper = torch.zeros(size)

        if self.classif.weight.is_cuda:
            warper = warper.cuda()

        warper[:x.size(0), :x.size(1), :x.size(2), :x.size(3)] = x

        return warper

    def forward(self, x):
        # Part 1
        l = self.lowpass(x)
        h = self.highpass(x)

        l = self.low1(l)
        h = self.high1(h)

        # Part 2
        tmp = self.warper(self.highpass(l), h.size())
        l = self.lowpass(l)

        h = torch.cat((h, tmp), 1)

        l = self.low2(l)
        h = self.high2(h)

        # Part 3
        '''
        tmp = self.warper(self.highpass(l), h.size())
        l = self.lowpass(l)
        
        h = torch.cat((h, tmp), 1)

        l = self.low3(l)
        h = self.high3(h)

        # Part 4
        tmp = self.warper(self.highpass(l), (l.size(0), l.size(1), h.size(2), h.size(3)))
        l = self.lowpass(l)
        
        h = torch.cat((h, tmp), 1)

        l = self.low4(l)
        h = self.high4(h)'''

        # Finale concat
        l = self.warper(l, (h.size(0), l.size(1), h.size(2), h.size(3)))
        y = torch.cat((h, l), 1)

        # Feature compression
        y = self.f_comp(y)
        y = self.f_comp2(y)
        y = self.f_comp3(y)
        y = y.view(y.size(0), -1)

        y = self.dropout(y)

        y = self.classif(y)

        return y

    def focal_trick(self, pi=0.01):
        list(self.parameters())[-1].data[0] = -torch.log(torch.tensor((1-pi)/pi))