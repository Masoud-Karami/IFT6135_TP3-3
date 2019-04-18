# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:52:14 2019

@author: karm2204
"""



#####         https://chrisorm.github.io/VAE-pyt.html
#             Variational Autoencoder in Pytorch
#             https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py
#             https://github.com/bobchennan/VAE_NBP/blob/master/vae_dp.py
#             https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.autograd import Variable
from sklearn.mixture import BayesianGaussianMixture
import torch.optim as optim
import torch.utils.data
import argparse



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
batch_size = 128
learning_rate = 31e-4
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 20


#%%

def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata

train, valid, test = get_data_loader("binarized_mnist", 64)

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
for x in train:
    plt.imshow(x[0, 0])
    break
    
#%%

parser = argparse.ArgumentParser(description='PyTorch VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
#parser.add_argument('--hidden',type=int,default=10,metavar='N',
#                    help='number of dimension for z')
#parser.add_argument('--comp',type=int,default=100,metavar='N',
#                    help='maximum number of components in DP')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#torch.manual_seed(args.seed)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size, shuffle=True, **kwargs)

#%%
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc_1 = nn.Linear(image_size, h_dim)
        self.fc_2 = nn.Linear(h_dim, z_dim)
        self.fc_3 = nn.Linear(h_dim, z_dim)
        self.fc_4 = nn.Linear(z_dim, h_dim)
        self.fc_5 = nn.Linear(h_dim, image_size)
        
        
    """ Encode a batch of samples, and return posterior parameters for each point."""    
    def encode(self, x):
        h_1 = F.relu(self.fc_1(x))
        return self.fc_2(h_1), self.fc_3(h_1)
    
    
    """ Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation.
        For each training sample (we get 128 batched at a time)
        - take the current learned mu, stddev for each of the z_dim 
        (in the pytorch VAE example, this is 20, z_dim = 20)
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians
        Parameters
        ----------
        mu : [128, z_dim] mean matrix
        logvar : [128, z_dim] variance matrix
        Returns
        -------
        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.
        """
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    """ Decode a batch of latent variables"""
    def decode(self, z):
        h_3 = F.relu(self.fc_4(z))
        return F.sigmoid(self.fc_5(h_3))
    
    
    """ Takes a batch of samples, encodes them, and then decodes them again to compare."""
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%

""" ELBO assuming entries of x are binary variables, with closed form KLD."""
def loss_function(x_reconst, x, mu, logvar):
    bce = F.binary_cross_entropy(x_reconst, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    # KLD /= x.view(-1, image_size).data.shape[0] * image_size
    return bce + KLD
#%%

# ----------
#  Train
# ----------
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. *
                batch_idx / len(train_loader), loss.item() / len(data)
            ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

#%%

# ----------
#  Test
# ----------
def test(epoch):
    model.eval()
    test_loss = 0
    # ind = np.arange(x.shape[0])
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
#           test_data = torch.from_numpy(test_data[np.random.choice(ind, size=batch_size)])
#           test_data = Variable(test_data, requires_grad=False)
            reconst_batch, mu, logvar = model(data)
            test_loss += loss_function(reconst_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        reconst_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#            print(data.view(batch_size, 2,2)[:n])
#            print(reconst_batch.view(batch_size, 2,2)[:n])
            
            
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
        
#%%    
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # 64 sets of random z_dim-float vectors, i.e. 64 locations / MNIST
        # digits in latent space
        with torch.no_grad():
            sample = torch.randn(64, z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), 'saved_models/vae.pth')            
            
            
            
            
            
