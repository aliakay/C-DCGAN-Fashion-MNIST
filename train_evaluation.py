from Model import*
from generate_test import*
from dataloader import load_data,load_real_images
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse

argparser = argparse.ArgumentParser(description='hyper-parameters')

argparser.add_argument('--nz',type=int, default=100, help='dimension of the latent space')
arg = argparser.parse_args()

# Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
#Generate random noise and labels for 10 image for each category
fixed_noise = torch.randn(100, arg.nz, 1, 1).to(device)
fixed_label = label_1hots[torch.arange(10).repeat(10).sort().values]

img_size=32 

def train_evaluation(img_size,fixed_noise,fixed_label):
    #See the fake and real images 
    # Size of the Figure
    plt.figure(figsize=(20,10))
    # Plot the real images
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title("Real Images")

    imgs=load_real_images()

    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0)*0.5+0.5)

    # Load the Best Generative Model

    netG=torch.load('Generator_model.pth',map_location=torch.device('cpu'))
    netG.eval()

    # Generate the Fake Images
    with torch.no_grad():
        fake = netG(fixed_noise.cpu(), fixed_label.cpu())

    # Plot the fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    fake = utils.make_grid(fake, nrow=10)
    plt.imshow(fake.permute(1, 2, 0)*0.5+0.5)

    # Save the comparation result
    plt.savefig('Assets/train_result_response.jpg', bbox_inches='tight')
    print("Real-Fake Images comparison picture saved as train_result_response.jpg")
    return 

train_evaluation(img_size,fixed_noise,fixed_label)
