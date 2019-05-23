import torch as ch
import itertools
import numpy as np
from matplotlib import cm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from models.simple_models import SmallSimpleClassifier
from cifar_config import trainloader, testloader, no_norm_loader, no_norm_testloader
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D

NET = ch.load(sys.argv[1])
NET.eval()
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SAVE_ITERS = 10

def encode(x):
    x = ch.stack([NORMALIZER(x[i]) for i in range(x.shape[0])])
    x = F.relu(NET.vae.conv_1(x))
    x = F.relu(NET.vae.conv_2(x))
    x = F.max_pool2d(x, 2)
    x = x.view(x.shape[0], -1)
    x = F.relu(NET.vae.fc_1(x))
    return NET.vae.fc_mean(x)/3.0

new_net = SmallSimpleClassifier().cuda()
new_net.load_state_dict(ch.load("results/trained_net_l2_flat"))
new_net.eval()
loss_fn = ch.nn.CrossEntropyLoss()

def visualize(orig_ims, correct_class, ind):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_marks = np.arange(-1.0, 1.0, 0.2)
    y_marks = np.arange(-10.0, 10.0, 2.0)
    xs, ys = np.meshgrid(x_marks, y_marks)
    orig_ims = orig_ims.detach()
    orig_ims.requires_grad = True
    loss = loss_fn(new_net(encode(orig_ims)), correct_class)
    g, = ch.autograd.grad(loss, orig_ims)
    g /= ch.norm(g)
    r = ch.randn_like(g).cuda()
    r /= ch.norm(r)
    zs = np.zeros_like(xs)
    for i, j in itertools.product(range(len(x_marks)), range(len(y_marks))):
        test_im = orig_ims + g * x_marks[i] + r * y_marks[j]
        loss = loss_fn(new_net(encode(test_im)), correct_class)
        zs[j][i] = loss
    ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm)
    plt.savefig("landscapes/test_%d.png" % (ind,))

num_correct = 0
num_total = 0
for (images, labels) in no_norm_testloader:
    images, labels = images.cuda(), labels.cuda()
    for i in range(images.shape[0]):
        visualize(images[i:i+1], labels[i:i+1], i)
    break
