#created by vignesh on 13/11/2017, 10am

import torch.nn as nn
from torch import nn
import torchvision as tv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
class Generator_model(nn.Module):
    def __init__(self):
        super(Generator_model,self).__init__()
        self.encoder()
        self.decoder()
    
    
            
    def encoder(self):
        self.lr_activation = nn.LeakyReLU(0.2)
        
        #3,256,256 - 64,128,128
        self.conv1 = nn.Conv2d(1,64,4,2,padding=1)
        
        #64,128,128 - 128,64,64
        self.conv2 = nn.Conv2d(64,128,4,2,padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        
        #128,64,64 - 256,32,32
        self.conv3 = nn.Conv2d(128,256,4,2,padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        
        #256,32,32 - 512,16,16
        self.conv4 = nn.Conv2d(256,512,4,2,padding=1)
        self.conv4_bn = nn.BatchNorm2d(512)
        #512,16,16 - 512,8,8 
        self.conv5 = nn.Conv2d(512,512,4,2,padding=1)
        self.conv5_bn = nn.BatchNorm2d(512)
        
        #512,8,8 - 512,4,4 
        self.conv6 = nn.Conv2d(512,512,4,2,padding=1)
        self.conv6_bn = nn.BatchNorm2d(512)
        
        #512,4,4 - 512,2,2
        self.conv7 = nn.Conv2d(512,512,4,2,padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)
        
        #512,2,2 - 512,1,1
        self.conv8 = nn.Conv2d(512,512,4,2,padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        #finally its 512,1,1
    
    def decoder(self):
        #512,1,1 - 512,2,2 
        self.deconv8 = nn.ConvTranspose2d(512,512,4,2,padding=1)
        self.deconv8_bn = nn.BatchNorm2d(512)
        
        #512+512,2,2 - 512,4,4 
        self.deconv7 = nn.ConvTranspose2d(512*2,512,4,2,padding=1)
        self.deconv7_bn = nn.BatchNorm2d(512)
        
        #512+512,4,4 - 512,8,8 
        self.deconv6 = nn.ConvTranspose2d(512*2,512,4,2,padding=1)
        self.deconv6_bn = nn.BatchNorm2d(512)
        
        #512+512,8,8 - 512,16,16 
        self.deconv5 = nn.ConvTranspose2d(512*2,512,4,2,padding=1)
        self.deconv5_bn = nn.BatchNorm2d(512)
        
        #512+512,16,16 - 256,32,32 
        self.deconv4 = nn.ConvTranspose2d(512*2,256,4,2,padding=1)
        self.deconv4_bn = nn.BatchNorm2d(256)
        
        #256+256,32,32 - 128,64,64
        self.deconv3 = nn.ConvTranspose2d(512,128,4,2,padding=1)
        self.deconv3_bn = nn.BatchNorm2d(128)
        
        #128+128,64,64 -64,128,128 
        self.deconv2 = nn.ConvTranspose2d(256,64,4,2,padding=1)
        self.deconv2_bn = nn.BatchNorm2d(64)
        
        #64+64,128,128 - 3,256,256
        self.deconv1 = nn.ConvTranspose2d(128,3,4,2,padding=1)
        self.deconv1_bn = nn.BatchNorm2d(3)
        
        
        
    def forward(self,x):
        #encoder
        a1=self.conv1(x)
        a1_lr=self.lr_activation(a1)
        
        a2=self.conv2(a1_lr)
        a2_bn=self.conv2_bn(a2)
        a2_lr=self.lr_activation(a2_bn)
        
        a3=self.conv3(a2_lr)
        a3_bn=self.conv3_bn(a3)
        a3_lr=self.lr_activation(a3_bn)
        
        a4=self.conv4(a3_lr)
        a4_bn=self.conv4_bn(a4)
        a4_lr=self.lr_activation(a4_bn)
                                 
        a5=self.conv5(a4_lr)
        a5_bn=self.conv5_bn(a5)
        a5_lr=self.lr_activation(a5_bn)
                                 
        a6=self.conv6(a5_lr)
        a6_bn=self.conv6_bn(a6)
        a6_lr=self.lr_activation(a6_bn)
                                 
        a7=self.conv7(a6_lr)
        a7_bn=self.conv7_bn(a7)
        a7_lr=self.lr_activation(a7_bn)
                                 
        a8=self.conv8(a7_lr)
        a8_bn=self.conv8_bn(a8)
        a8_lr=self.lr_activation(a8_bn)
        #decoder                        
        d8 = self.deconv8(a8_lr)
        d8_bn=self.deconv8_bn(d8)
        d8_dropout=nn.functional.dropout(d8_bn)
        d8_unet=torch.cat((d8_dropout, a7_lr), 1)
        
        d7=self.deconv7(d8_unet)
        d7_bn=self.deconv8_bn(d7)
        d7_dropout=nn.functional.dropout(d7_bn)
        d7_unet=torch.cat((d7_dropout, a6_lr), 1)
                                 
        d6=self.deconv6(d7_unet)
        d6_bn=self.deconv6_bn(d6)
        d6_dropout=nn.functional.dropout(d6_bn)
        d6_unet=torch.cat((d6_dropout, a5_lr), 1)
            
        d5=self.deconv5(d6_unet)
        d5_bn=self.deconv5_bn(d5)
        d5_unet=torch.cat((d5_bn, a4_lr), 1)
                                 
        d4=self.deconv4(d5_unet)
        d4_bn=self.deconv4_bn(d4)
        d4_unet=torch.cat((d4_bn, a3_lr), 1)
        
        d3=self.deconv3(d4_unet)
        d3_bn=self.deconv3_bn(d3)
        d3_unet=torch.cat((d3_bn, a2_lr), 1)
            
        d2=self.deconv2(d3_unet)
        d2_bn=self.deconv2_bn(d2)
        d2_unet=torch.cat((d2_bn, a1_lr), 1)    
        
        d1=self.deconv1(d2_unet)
        d1_bn=self.deconv1_bn(d1)
        img=nn.functional.tanh(d1_bn)
        return img
     
class Discriminator_Model(nn.Module):
        def __init__(self):
            super(Discriminator_Model, self).__init__()
            self.conv1 = nn.Conv2d(3+1,64,4,2,padding=1)
            
            self.conv2= nn.Conv2d(64,128,4,2,padding=1)
            self.conv2_bn = nn.BatchNorm2d(128)
            
            self.conv3= nn.Conv2d(128,256,4,2,padding=1)
            self.conv3_bn = nn.BatchNorm2d(256)
            
            self.conv4= nn.Conv2d(256,512,4,1,padding=1)
            self.conv4_bn = nn.BatchNorm2d(512)
            
            self.conv5= nn.Conv2d(512,1,4,1,padding=1)
            
        def forward(self,input,output):
            a1=self.conv1(torch.cat([input,output],1))
            a1_activ=F.leaky_relu(a1,0.2)
            a2=self.conv2_bn(self.conv2(a1_activ))
            a2_activ=F.leaky_relu(a2,0.2)
            a3=self.conv3_bn(self.conv3(a2_activ))
            a3_activ=F.leaky_relu(a3,0.2)
            a4=self.conv4_bn(self.conv4(a3_activ))
            a4_activ=F.leaky_relu(a4,0.2)
            a5=self.conv5(a4_activ)
            prob=F.sigmoid(a5)
            return prob
        
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class losses():
    def __init__(self):
        self.ones  = torch.ones(100).cuda()
        self.zeros = torch.zeros(100).cuda()
        self.ones=Variable(self.ones)
        self.zeros=Variable(self.zeros)
    
    def dloss(self,real,fake):
        self.ones.data.resize_(real.size()).fill_(1)
        self.zeros.data.resize_(real.size()).fill_(0)
        return F.binary_cross_entropy(real,self.ones)+F.binary_cross_entropy(fake,self.zeros)
        
    def gloss(self,G_fake,D_fake,color):
        return F.binary_cross_entropy(D_fake,self.ones)+100*F.smooth_l1_loss(G_fake,color)
        

