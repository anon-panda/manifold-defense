import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from models.simple_models import SmallSimpleClassifier
from cifar_config import trainloader, testloader, no_norm_loader, no_norm_testloader
from torchvision import transforms
from attacks import pgd_l2, pgd_linf
from argparse import ArgumentParser

NET = ch.load(sys.argv[1])
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def encode(x):
    x = ch.stack([NORMALIZER(x[i]) for i in range(x.shape[0])])
    x = F.relu(NET.vae.conv_1(x))
    x = F.relu(NET.vae.conv_2(x))
    x = F.max_pool2d(x, 2)
    x = x.view(x.shape[0], -1)
    x = F.relu(NET.vae.fc_1(x))
    return NET.vae.fc_mean(x)/3.0

label_pairs_dict = {}
for i, (images, labels) in enumerate(no_norm_loader):
    # Shape of images: (BATCH_SIZE x channels x width x height)
    # Shape of labels: (BATCH_SIZE)
    images, labels = images.cuda(), labels.cuda()
    new_ims = encode(images.clone())
    for j in range(len(images)):
        for k in range(j):
            label_j = labels[j].item()
            label_k = labels[k].item()
            if (label_j, label_k) not in label_pairs_dict:
                label_pairs_dict[(label_j, label_k)] = []
            dist = ch.norm(new_ims[j] - new_ims[k])
            label_pairs_dict[(label_j, label_k)].append(dist)
    for key in label_pairs_dict:
        if (key[1], key[0]) in label_pairs_dict:
            label_pairs_dict[key] = label_pairs_dict[key] + label_pairs_dict[(key[1], key[0])]
    avgd = {k: ch.mean(ch.stack(label_pairs_dict[k])) for k in label_pairs_dict}
    print(avgd)
    label_pairs_dict = {}
