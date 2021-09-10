from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from efficientnet_pytorch.utils import get_same_padding_conv2d,round_filters


class efficientNet_gru(nn.Module):
    def __init__(self,in_channels=4,out_channels=2,hidden_channels=100,image_size=256,length=32):
        super(efficientNet_gru,self).__init__()


        '''base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[1:-1]

        self.base_layers = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),*layers,nn.Conv2d(512,512,kernel_size=16,stride=1))  # block5_conv3 output
        self.bgru=nn.GRU(512,64,bidirectional=True,batch_first=True)
        self.classifier=nn.Linear(length*128,out_channels,bias=True)'''
        model = EfficientNet.from_name('efficientnet-b0')
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        model._conv_stem = Conv2d(in_channels, 32, kernel_size=3, stride=2, bias=False)
        feature = model._fc.in_features
        model._fc = nn.Linear(in_features=feature, out_features=hidden_channels, bias=True)
        self.base_layers = model
        self.bgru = nn.GRU(hidden_channels, 64, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(length * 128, out_channels, bias=True)


    def forward(self,x):
        bs,inchannels,imgsize,_,length=x.size()
        x=x.permute(0,4,2,3,1).contiguous().reshape(bs*length,imgsize,imgsize,inchannels)

        x=x.permute(0,3,1,2).contiguous()#bs*length,inchannels,imgsize,imgsize
        x=self.base_layers(x)#bs*length,outchannels
        outchannels=x.size()[1]
        x=x.view(bs,length,outchannels)

        x,_=self.bgru(x)
        x=x.reshape(bs,-1)

        return self.classifier(x)