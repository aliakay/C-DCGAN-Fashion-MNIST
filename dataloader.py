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
# scale an array of images to a new size
from skimage.transform import resize
import numpy as np
from numpy import asarray

root_folder = "FASHION_MNIST"

img_size=32

def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.uint8(images_list)
 

import PIL.Image as Image

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        resized_img.convert('RGB')
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

def load_data(batch_size,img_size):
    # create a set of transforms for the dataset
    dset_transforms = list()
    dset_transforms.append(transforms.Resize(img_size))
    dset_transforms.append(transforms.ToTensor())
    dset_transforms.append(transforms.Normalize(
                                    [0.5], [0.5]))
    dset_transforms = transforms.Compose(dset_transforms)

    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root = root_folder,
        train = True,
        download = True,
        transform=dset_transforms
        )

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return data_loader

def load_real_images():
    dset_transforms = list()
    dset_transforms.append(transforms.Resize(img_size))
    dset_transforms.append(transforms.ToTensor())
    dset_transforms.append(transforms.Normalize(
                                    [0.5], [0.5]))
    dset_transforms = transforms.Compose(dset_transforms)

    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root = root_folder,
        train = True,
        download = True,
        transform=dset_transforms
        )

    imgs = {}
    for x, y in train_set:
        if y not in imgs:
            imgs[y] = []
        elif len(imgs[y])!=10:
            imgs[y].append(x)
        elif sum(len(imgs[key]) for key in imgs)==100:
            break
        else:
            continue
                
    imgs = sorted(imgs.items(), key=lambda x:x[0])
    imgs = [torch.stack(item[1], dim=0) for item in imgs]
    imgs = torch.cat(imgs, dim=0)
    return imgs
