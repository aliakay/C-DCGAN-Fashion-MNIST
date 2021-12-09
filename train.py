# import libraries
from Model import weights_init,Generator,Discriminator
from generate_test import create_noise_label,Generate_samples
from dataloader import load_data,load_real_images,scale_images,interpolate
from Frechet_distance import InceptionV3,calculate_activation_statistics,calculate_frechet_distance,calculate_fretchet

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

import argparse

argparser = argparse.ArgumentParser(description='hyper-parameters')

argparser.add_argument('--batch_size', type=int,   default=128,   help='batch size')
argparser.add_argument('--ngf',        type=int,   default=32,    help='number of features in the generator')
argparser.add_argument('--ndf',        type=int,   default=32,    help='number of channels in the discriminator')
argparser.add_argument('--nz',         type=int,   default=100,   help='dimension of the latent space')
argparser.add_argument('--num_epochs', type=int,   default=50,    help='number of training epoch')
argparser.add_argument('--nc',         type=int,   default=1,     help='number of channels in an image (RGB)')
argparser.add_argument('--lr',         type=float, default=0.0002,help='learning rate')

arg = argparser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device type: ",device)

n_class=10
img_size=32 

data_loader=load_data(batch_size=arg.batch_size,img_size=img_size)

# Create the generator
netG = Generator(arg.nz,arg.ngf,arg.nc).to(device)
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(arg.nc,arg.ndf).to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# import Pre-trained InceptionV3 network for Frechet Distance
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.to(device)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
#---- Label Smoothing-----
real_label_num = 0.9
fake_label_num = 0.1

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=arg.lr, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=arg.lr, betas=(0.5, 0.999))

# Label one-hot for G
label_1hots = torch.zeros(10,10)
for i in range(10):
    label_1hots[i,i] = 1
label_1hots = label_1hots.view(10,10,1,1).to(device)

# Label one-hot for D
label_fills = torch.zeros(10, 10, img_size, img_size)
ones = torch.ones(img_size, img_size)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

# Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
fixed_noise = torch.randn(arg.batch_size, arg.nz, 1, 1).to(device)
fixed_label = label_1hots[torch.randint(0, 10, (arg.batch_size,)).sort().values]


# Lists to keep track of progress
img_list = []
losses=[]
D_x_list = []
D_z_list = []
best=np.inf
iters = 0

writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
writer0 = SummaryWriter(f"logs/GAN_MNIST/loss")


print("Starting Training Loop...")
# For each epoch
for epoch in range(arg.num_epochs):

    # For each batch in the dataloader
    for i, data in enumerate(data_loader):

        ############################
        # (1) Update D network: maximize log(D(x|y)) + log(1 - D(G(z|y)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        
        # Format batch
        real_image = data[0].to(device)
        b_size = real_image.size(0)

        real_label = torch.full((b_size,), real_label_num).to(device)
        fake_label = torch.full((b_size,), fake_label_num).to(device)
        
        G_label = label_1hots[data[1]]
        D_label = label_fills[data[1]]
        
        # Forward pass real batch through D
        output = netD(real_image, D_label).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, real_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, arg.nz, 1, 1).to(device)
        # Generate fake image batch with G
        fake = netG(noise, G_label)
        # Classify all fake batch with D
        output = netD(fake.detach(), D_label).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, fake_label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z|y)))
        ###########################
        netG.zero_grad()
        
        output = netD(fake, D_label).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, real_label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # print the training losses
        if iters % 100 == 0:
          print('[%3d/%d][%3d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, 
              arg.num_epochs, i, len(data_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
          writer0.add_scalars('Discrimitor-Generator Losses', {'Discrimitor Loss':errD.item(),
                                    'Generator':errG.item()}, iters)
          

        if (iters % 469 == 0):
          losses.append((errD.item(), errG.item())) 
          # Add values to plots
          netG.eval()  
          with torch.no_grad():
            fake_fixed = netG(fixed_noise,fixed_label).cpu()
            img_list.append(vutils.make_grid(fake_fixed, padding=2, normalize=True))

            img_grid_real = torchvision.utils.make_grid(real_image[:64], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake_fixed[:64], normalize=True)

            writer_real.add_image("Real", img_grid_real, global_step=iters)
            writer_fake.add_image("Fake", img_grid_fake, global_step=iters)
            # display the images inline as well
            #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            #ax.imshow(np.transpose(vutils.make_grid(fake_fixed, padding=2, normalize=True), (1, 2, 0)))
            #ax.axis('off')
            #ax.set_title('Fixed Noise Samples')
            #plt.show()

            fake_fixed = scale_images(fake_fixed, (299,299,3))
            # change the real_images' shape too - so that it keep matching
            # fake_images' shape.
            real_image = scale_images(real_image.cpu(), (299,299,3))

            fake_fixed=interpolate(fake_fixed)
            real_image=interpolate(real_image)

            fretchet_dist=calculate_fretchet(real_image.to(device),fake_fixed.to(device),model) 
            print("Fretchet Score",fretchet_dist)
            writer0.add_scalar('Fretchet Score', fretchet_dist,iters)

            PATH = "Generator_model_best_FID_score.pth"
            if fretchet_dist < best:
              best = fretchet_dist
              torch.save(netG, PATH)
              print("model saved.")
        iters += 1

torch.save(netG, "Generator_model.pth")
print("Last model saved.")

writer_real.close()
writer_fake.close()
writer0.close()

