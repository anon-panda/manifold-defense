import torch as ch
import itertools
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as snorm
import numpy as np
import math

class IdentityEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args,**kwargs):
        return x

'''
class MNISTVAE(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super(MNISTVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, embed_feats)
        self.fc22 = nn.Linear(400, embed_feats)
        self.fc3 = nn.Linear(embed_feats, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500,784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return mu + eps*std

    def decode(self, z,square=True):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        if square:
            return ch.sigmoid(self.fc5(h4)).view(-1,1,28,28)
        else:
            return ch.sigmoid(self.fc5(h4))

    def forward(self, x, latent=False, square=True):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if latent:
            return self.decode(z,square), mu, logvar
        else:
            return self.decode(z,square)
'''
'''
class MNISTVAE(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super(MNISTVAE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding=0, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc_mu = nn.Linear(num_feats*8, embed_feats)
        self.fc_sigma = nn.Linear(num_feats*8, embed_feats)
        
        self.fc_decode_1 = nn.Linear(embed_feats, 3136)
        self.deconv_decode_1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.deconv_decode_2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv_decode_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=1)


    def encode(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        return self.fc_mu(out), self.fc_sigma(out)
        

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return mu + eps*std

    def decode(self, z,square=True):
        out = F.relu(self.fc_decode_1(z))
        out = out.view(out.shape[0], 16, 14, 14)
        out = F.relu(self.deconv_decode_1(out))
        out = F.relu(self.deconv_decode_2(out))
        out = self.conv_decode_1(out)
        if square:
            return ch.sigmoid(out).view(-1,1,28,28)
        else:
            return ch.sigmoid(out)

    def forward(self, x, latent=False, square=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if latent:
            return self.decode(z,square), mu, logvar
        else:
            return self.decode(z,square)
'''

#'''
class MNISTVAE(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True, leaky_relu=True):
        super(MNISTVAE, self).__init__()

        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, 20)
        self.fc22 = nn.Linear(500, 20)
        self.fc3 = nn.Linear(20,500)#, bias=False)
        self.fc4 = nn.Linear(500,500)
        self.fc5 = nn.Linear(500, 784)#, bias=False)
        
        self.leaky_relu = leaky_relu
        if leaky_relu:
            self.nonlinearity = nn.LeakyReLU(0.1)
        else:
            self.nonlinearity = nn.ReLU()

    def encode(self, x):
        h1 = self.nonlinearity(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = ch.exp(0.5*logvar)
        eps = ch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z,square=True):
        out = self.nonlinearity(self.fc3(z))
        out = self.nonlinearity(self.fc4(out))
        if square:
            return ch.sigmoid(self.fc5(out)).view(-1,1,28,28)
        else:
            return ch.sigmoid(self.fc5(out))

    def forward(self, x,latent=False):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if latent:
             return self.decode(z), mu, logvar
        else:
             return self.decode(z)
#'''

class MNISTAutoEncoder(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super().__init__()
        if spectral_norm:
            self.conv1 = snorm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1))
            self.conv2 = snorm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
            self.fc = snorm(nn.Linear(2*3136, embed_feats))
            self.fc_decode_1 = snorm(nn.Linear(embed_feats, 3136))
            self.deconv_decode_1 = snorm(nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2))
            self.conv_decode_1 = snorm(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2))
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding=0, stride=2)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.fc = nn.Linear(num_feats*8, embed_feats)
            self.fc_decode_1 = nn.Linear(embed_feats, 3136)
            self.deconv_decode_1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.deconv_decode_2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            self.conv_decode_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=1)

        self.no_decode = no_decode
        self.sn = spectral_norm
        self.num_feats = num_feats
        
        decode_layers = [self.fc_decode_1, self.deconv_decode_1, self.deconv_decode_2, self.conv_decode_1]
        self.decode_vars = itertools.chain(*[x.parameters() for x in decode_layers])

    def encode(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        return ch.tanh(self.fc(out))

    def decode(self, z):
        out = F.relu(self.fc_decode_1(z))
        out = out.view(out.shape[0], 16, 14, 14)
        out = F.relu(self.deconv_decode_1(out))
        out = F.relu(self.deconv_decode_2(out))
        out = self.conv_decode_1(out)
        return (ch.tanh(out) + 1)/2

    def forward(self, x, no_decode=False):
        z = self.encode(x)
        if self.no_decode or no_decode:
            return z
        d = self.decode(z)
        d = d.view(d.shape[0], 1, 28, 28)
        return d

class CIFARAutoEncoder(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super().__init__()
        self.convs = []
        if spectral_norm:
            raise NotImplementedError
        else:
            self.conv_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.conv_2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=0, stride=2)
            self.bn_enc_1 = nn.BatchNorm2d(32)
            self.conv_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.bn_enc_2 = nn.BatchNorm2d(32)
            self.conv_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            self.bn_enc_3 = nn.BatchNorm2d(32)
            self.fc_1 = nn.Linear(8192, 128)

            self.fc_mean = nn.Linear(128, embed_feats)

        self.fc_decode_1 = nn.Linear(embed_feats, 128)
        self.fc_decode_2 = nn.Linear(128, 8192)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_dec_1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_dec_2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.bn_dec_3 = nn.BatchNorm2d(32)
        self.conv_decode_1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

        self.no_decode = no_decode

        decode_layers = [self.fc_decode_1, self.fc_decode_2, self.deconv1, self.deconv2, self.deconv3, self.conv_decode_1]
        self.decode_vars = itertools.chain(*[x.parameters() for x in decode_layers])

    def encode(self, x):
        out = F.relu(self.conv_1(x))
        out = F.relu(self.conv_2(out))
        out = self.bn_enc_1(out)
        out = F.relu(self.conv_3(out))
        out = self.bn_enc_2(out)
        out = F.relu(self.conv_4(out))
        out = self.bn_enc_3(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc_1(out))
        return self.fc_mean(out)

    def decode(self, z):
        dec = F.relu(self.fc_decode_1(z))
        dec = F.relu(self.fc_decode_2(dec))
        dec = dec.view(dec.shape[0], 32, 16, 16)
        dec = F.relu(self.deconv1(dec))
        dec = self.bn_dec_1(dec)
        dec = F.relu(self.deconv2(dec))
        dec = self.bn_dec_2(dec)
        dec = F.relu(self.deconv3(dec))
        dec = self.bn_dec_3(dec)
        dec = self.conv_decode_1(dec)
        return (ch.tanh(dec)+1)/2

    def forward(self, x, no_decode=False):
        z = self.encode(x)
        if self.no_decode or no_decode:
            return F.relu(z)
        return self.decode(z)

#'''
class CIFARGenerator(ch.nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True, leaky_relu=True):
        super(CIFARGenerator,self).__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def decode(self, x):
        x = self.main_module(x)
        return 0.5*self.output(x) + 0.5
#'''
'''
class CIFARGenerator(nn.Module):
    def __init__(self, num_feats, embed_feats, no_decode=False, spectral_norm=True):
        super(CIFARGenerator, self).__init__()
        
        n = math.log2(np.sqrt(num_feats//3))
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        ngf = 64 
        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(embed_feats, ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid{0}-{1}conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid{0}batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid{0}relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf,3, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())


    def decode(self, input):
        output = self.decoder(input)
        return 0.5*output + 0.5
    
    def make_cuda(self):
        self.decoder.cuda()
'''
