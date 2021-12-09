
from Model import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image
from PIL import Image
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

img_size=32

argparser = argparse.ArgumentParser(description='hyper-parameters')

argparser.add_argument('--batch_size', type=int,   default=128,   help='batch size')
argparser.add_argument('--ngf',        type=int,   default=32,    help='number of features in the generator')
argparser.add_argument('--ndf',        type=int,   default=32,    help='number of channels in the discriminator')
argparser.add_argument('--nz',         type=int,   default=100,   help='dimension of the latent space')
argparser.add_argument('--num_epochs', type=int,   default=10,    help='number of training epoch')
argparser.add_argument('--nc',         type=int,   default=1,     help='number of channels in an image (RGB)')
argparser.add_argument('--lr',         type=float, default=0.0002,help='learning rate')
arg = argparser.parse_args()

idx_to_label = {
    'T-shirt/top': 0,
    'Trouser': 1,
    'Pullover': 2,
    'Dress': 3,
    'Coat': 4,
    'Sandal': 5,
    'Shirt': 6,
    'Sneaker': 7,
    'Bag': 8,
    'Ankle boot': 9
    }

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

def create_noise_label(batch_size,nz):
    z = torch.randn(batch_size, nz, 1, 1).to(device)
    t_test = torch.randint(0, 10, (batch_size,)).sort().values
    n = 0
    for i in range(100):
        t_test[i] = n
        if (i+1)%10 == 0:
            n += 1
    return z,t_test

def Generate_samples(category):
      # Labels for test.  
    # import pre-trained the generator
    netG=torch.load("Generator_model.pth",map_location=torch.device('cpu'))
    netG.eval()
    #Generate sample noise
    z,t_test = create_noise_label(arg.batch_size,arg.nz)
    label_1hot = label_1hots[t_test]
    nx = t_test.numpy()
    #match with given category
    index=np.where(nx == idx_to_label[category])[0][0]

    sample_images = netG(z.cpu(), label_1hot.cpu()).data.cpu()
    #plt.imshow(sample_images[index].squeeze(0),cmap="gray")
    #plt.savefig('test1.png')
    img1 = sample_images[index].squeeze(0)
    save_image(img1,'Assets/'+category+"test.png")


