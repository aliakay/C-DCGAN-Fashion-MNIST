# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.autograd import Variable
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self,nz,ngf,nc):
        super(Generator, self).__init__()
      
        self.image = nn.Sequential(
            
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            
        )
        self.label = nn.Sequential(
            
            nn.ConvTranspose2d(num_classes, ngf * 8, 4, 1,0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
            
        )
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)


class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator, self).__init__()

        self.image = nn.Sequential(
           
            nn.Conv2d(nc, ndf, 4, 2, 1,1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
            
        )
        self.label = nn.Sequential(
            
            nn.Conv2d(num_classes, ndf, 4, 2, 1,1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
            
        )
        self.main = nn.Sequential(
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            
            nn.Sigmoid()
        )

    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)

