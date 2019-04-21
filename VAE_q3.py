# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:05:29 2019

@author: karm2204
"""

"""
References:
"""
#%%

# https://towardsdatascience.com/model-summary-in-pytorch-b5a1e4b64d25
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730
# https://discuss.pytorch.org/t/understanding-output-padding-cnn-autoencoder-input-output-not-the-same/22743
# https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
# https://github.com/pytorch/examples/blob/master/vae/main.py
# https://www.groundai.com/project/isolating-sources-of-disentanglement-in-variational-autoencoders/
# https://arogozhnikov.github.io/einops/pytorch-examples.html

# Note   :    https://www.cs.toronto.edu/~lczhang/360/lec/w03/convnet.html
#%%

import argparse
import torch
import torch.nn as nn 
from torch import cuda
import torch.utils.data
from torch import optim, autograd
from torch.nn import functional as F
import torchvision
from torch.utils.data import dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

#%%
parser = argparse.ArgumentParser(description='TP3 #3 PyTorch VAE for SVHN dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--latent_dim', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
# parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
# parser.add_argument('--logvar', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument("--sample_dir", type=str, default="samples", help="Directory containing samples for"
parser.add_argument("--save_path", type=str, default="VAE_q3.pt")                                                                      
parser.add_argument("--load_path", type=str, default="VAE_q3.pt")
parser.add_argument("-s_true", action="store_true", help="Flag to specify if we train the model")

# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730
args = parser.parse_args()
args.device = torch.device("cuda") if cuda.is_available() else torch.device('cpu')

#%%    
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])
    
def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader 
                    
                    
#%%
def imshow(img):
    img = 0.5*(img + 1)
    npimg = img.numpy()
    # npimg = (255*npimg).astype(np.uint8) # to be a int in (0,...,255)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()                 
                    
#%% 
class View(nn.Module):
    def __init__(self, shape, *shape_):
        super().__init__()
        if isinstance(shape, list):
            self.shape = shape
        else:
            self.shape = (shape,) + shape_      
            
def forward(self, x):
        return x.view(self.shape)
                    
#%%                
# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730
                    
class VAE(nn.Module):
    def __init__(self, batch_size, latent_dim=100):
        super(VAE, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.convencoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            #  Layer 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Layer 3
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Layer 4
            View(self.batch_size, 4*4*512),
            nn.Linear(4*4*512, 2*self.latent_dim)
        )

        self.convdecoder = nn.Sequential(
            # Layer 1
            nn.Linear(self.latent_dim, 4*4*512),
            View(self.batch_size, 512, 4, 4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Layer 2
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Layer 3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # Layer 4
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )


    def forward(self, x):
        convencoder = self.convencoder(x)
        mu, logvar = convencoder[:, :self.latent_dim], convencoder[:, self.latent_dim:]
        z = torch.randn_like(mu, device = args.device)
        f_x = mu + torch.exp(logvar) * z
        decode_z = self.convdecoder(f_x)
        return mu, logvar, decode_z

#%% 
def kl_div(mu, logvar):
    return 0.5 * (-1. - 2.*logvar + torch.exp(logvar)**2. + mu**2.).sum(dim=1)


def log_like(x, x_):
    k = x.size()[1]
    return -k/2 * torch.log(2 * np.pi * torch.ones(1)) -0.5 * ((x - x_)**2.).mean(dim=1)


def train_model(model, train, valid, save_path):
    adam = optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(20):
        for batch, i in train:
            # put batch on device
            batch = batch.to(args.device)

            # obtain the parameters from the encoder and compute KL divergence
            mu, logvar, decode_z = model(batch)
            kl = kl_div(mu, logvar)

            # compute the reconstruction loss
            logpx_z = log_like(batch.view(-1, 3*32*32), decode_z.view(-1, 3*32*32))

            # combine the two loss terms and compute gradients
            elbo = (logpx_z - kl).mean()

            # maximize the elbo i.e. minimize - elbo
            autograd.backward([-elbo])

            # Update the parameters and zero the gradients for the next mini-batch
            adam.step()
            adam.zero_grad()

        # compute the loss for the validation set
        valid_elbo = torch.zeros(1)
        nb_batches = 0
        for batch, i in valid:
            nb_batches += 1
            mu, log_sigma, decode_z = model(batch)
            kl = kl_div(mu, log_sigma)
            logpx_z = log_like(valid.view(-1, 3*32*32), decode_z.view(-1, 3*32*32))
            valid_elbo += (logpx_z - kl).mean()
        valid_elbo /= nb_batches
        print("After epoch {} the validation loss is: ".format(epoch+1), valid_elbo.item())

    # save the model to be used later
    torch.save(model.state_dict(), save_path)        
# https://www.groundai.com/project/isolating-sources-of-disentanglement-in-variational-autoencoders/
                    
#%%        
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    args.device = device
    train, valid, test = get_data_loader("svhn", batch_size = 64)
    model = VAE(batch_size = args.batch_size, latent_dim = args.latent_dim)   
    # show images
    imshow(torchvision.utils.make_grid(images))
    dataiter = iter(train)
    images, labels = dataiter.next()
    print( labels[0] )

    if parser.parse_args().s_true:
        train_model(model, train, valid, args.save_path)
    else:
        model.load_state_dict(torch.load(args.load_path))
        model.eval()                
                    
                    
                    
                    
                    
                    
                    
                    
                    

