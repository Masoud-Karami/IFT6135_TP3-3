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
# pip install torchsummary

# Note   :    https://www.cs.toronto.edu/~lczhang/360/lec/w03/convnet.html
#%%

import argparse
import torch
import torch.nn as nn 
from torch import cuda
import torch.utils.data
from torch import optim, autograd
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from classify_svhn import get_data_loader
#from torchsummary import summary
#import os

#%%
parser = argparse.ArgumentParser(description='TP3 #3 PyTorch VAE for SVHN dataset')
parser.add_argument("--model_path", type=str,help="The path for the model checkpoint")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--logvar', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument("--sample_dir", type=str, default="samples", help="Directory containing samples for"
#parser.add_argument("--save_path", type=str, default="vae_q3.pt")                                                                      "evaluation")

#%%

# https://discuss.pytorch.org/t/text-autoencoder-nan-loss-after-first-batch/22730

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim=100):
        super(ConvEncoder, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.conv_1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.pool_2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool_4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv_5 = nn.Conv2d(64, 256, kernel_size=(7, 7), padding=0)
        self.fc_6_1 = nn.Linear(input_shape[1] * input_shape[2], latent_dim)
        self.fc_6_2 = nn.Linear(input_shape[1] * input_shape[2], latent_dim)

    def forward(self, x):
        h_1 = F.elu(self.conv_1(x))
        h_2 = self.pool_2(h_1)
        h_3 = F.elu(self.conv_3(h_2))
        h_4 = self.pool_4(h_3)
        h_5 = F.elu(self.conv_5(h_4))
        h_5 = h_5.view(h_5.size(0), -1)
        return self.fc_6_1(h_5), self.fc_6_2(h_5)
#%%
        
class ConvDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim=100):
        super(ConvDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.fc_7 = nn.Linear(latent_dim, 256, bias=True)
        self.conv_8 = nn.Conv2d(256, 64, kernel_size=(5, 5), padding=4)
        self.up_9 = nn.UpsamplingBilinear2d(scale_factor=2, mode='bilinear')
        self.conv_10 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2)
        self.up_11 = nn.UpsamplingBilinear2d(scale_factor=2, mode='bilinear')
        self.conv_12 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=2)
        self.conv_13 = nn.Conv2d(16, output_shape[0], kernel_size=(3, 3), padding=2)

    def forward(self, z):
        h_6 = F.elu(self.fc_7(z))
        h_6 = h_6.view(h_6.size(0), h_6.size(1), 1, 1)
        h_7 = F.elu(self.conv_8(h_6))
        h_8 = self.up_9(h_7)
        h_9 = F.elu(self.conv_10(h_8))
        h_10 = self.up_11(h_9)
        h_11 = F.elu(self.conv_12(h_10))
        out = F.sigmoid(self.conv_13(h_11))
        return out

#%%
        
# https://www.groundai.com/project/isolating-sources-of-disentanglement-in-variational-autoencoders/
        
class MLPEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim=100, hidden_layer_size=400):
        super(MLPEncoder, self).__init__()

        self.input_shape = input_shape
        self.flat_dim = input_shape[1] * input_shape[2]
        self.latent_dim = latent_dim

        self.fc_1 = nn.Linear(self.flat_dim, hidden_layer_size)
        self.fc_2 = nn.Linear(hidden_layer_size, latent_dim)
        self.fc_3 = nn.Linear(hidden_layer_size, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.input_shape[0], self.flat_dim)
        h_1 = F.relu(self.fc_1(x))
        return self.fc_2(h_1), self.fc_3(h_1)
#%%
        
class MLPDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim=100, hidden_layer_size=400):
        super(MLPDecoder, self).__init__()

        self.output_shape = output_shape
        self.flat_dim = output_shape[1] * output_shape[2]
        self.latent_dim = latent_dim

        self.fc3 = nn.Linear(latent_dim, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, self.flat_dim)

    def forward(self, z):
        h_3 = F.relu(self.fc_3(z))
        out = torch.sigmoid(self.fc4(h_3))
        return out.view(-1, self.output_shape[0], self.output_shape[1], self.output_shape[2])

#%%
        
class VAE(nn.Module):
    def __init__(self, encoder, decoder, n_channels=3):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.latent_dim == decoder.latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

#%%
        
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#%%
    
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
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

#%%
    
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_{}.png'.format(epoch), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


#%%

#def evaluation(model):

#%%


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./data', split='test', transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # VAE model (TODO: change hardcoded shapes)
    model = VAE(encoder=ConvEncoder(input_shape=(3, 32, 32)),
                decoder=ConvDecoder(output_shape=(3, 32, 32)))

    train_model = True
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        train_model = False
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    best_loss = float('inf')
    patience = 0


    RANDOM_SAMPLE = True
    PERTURBATED_SAMPLE = False
    INTERPOLATED_SAMPLE = False
    EPS = 0.01

    for epoch in range(1, args.epochs + 1):
        if train_model:
            train_loss = train(epoch)

        test_loss = test(epoch)

        with torch.no_grad():
            # Random sample from z
            if RANDOM_SAMPLE:
                z = torch.randn(128, model.decoder.latent_dim)
            # Small perturbation in each dimension
            if PERTURBATED_SAMPLE:
                z = torch.ones(128, 1, model.decoder.latent_dim) * EPS
            # Interpolation between two points
            if INTERPOLATED_SAMPLE:
                z = torch.ones(128, 1, model.decoder.latent_dim)

            z = z.to(device)
            sample = model.decode(z).cpu()
            save_image(sample, 'results/sample_' + str(epoch) + '.png')

        if test_loss < best_loss:
            # save model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'results/model.pt')
            best_loss = test_loss
            patience = 0
        elif patience <= 3:
            patience += 1
            print('Patience {}'.format(patience))
        else:
            print('Early stopping after {} epochs'.format(epoch))
            break