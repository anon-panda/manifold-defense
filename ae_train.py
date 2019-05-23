import torch as ch
import scipy
from scipy.misc import imsave
import json
from torchvision import transforms # Image processing
from torchvision import models as m
from torchvision import datasets # Auto downloading and processing datasets
from torch import nn # Neural networks
import torch.nn.functional as F # Neural network utilities
import torch.optim as optim # Optimizers
from models import resnet
from argparse import ArgumentParser
from attacks import norm_sep_attack, decoder_norm_sep_attack

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64,
        help="Batch size for training")
parser.add_argument("-d", "--dataset", choices=["mnist", "cifar"], required=True,
        help="Which dataset to use")
parser.add_argument("-o", "--output-name", required=True, type=str,
        help="Where to save the trained network")
parser.add_argument("--embed-feats", type=int, default=128,
        help="Size of the latent space")
parser.add_argument("--num-epochs", type=int, default=400,
        help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.01,
        help="Learning rate for training")
parser.add_argument("-s", "--save-epochs", default=10,
        help="How often to save the model (every $s epochs)")
parser.add_argument("--use-dropout", action="store_true",
        help="Whether to use Dropout in training")
parser.add_argument("--reconstruction-loss", default=-1, type=float,
        help="Coeffecient of reconstruction loss in VAE (-1 = no reconstruction loss)")
parser.add_argument("--lr-decay", default=0.2, type=float,
        help="Rate at which to decay the learning rate")
parser.add_argument("--at", type=float,
        help="If given, use adv training to enforce Lipschitz constant with coefficient")
parser.add_argument("--at-dec", type=float,
        help="If given, use adv training to enforce DECODER is Lipschitz too")
parser.add_argument("--at-ns", type=int, default=10,
        help="Number of steps to use for adv training")
parser.add_argument("--at-eps", type=float,
        help="Used with the previous option, epsilon for adv training")
parser.add_argument("--vae-loss", action="store_true", 
        help="If specified, then also make the latent space be mean-zero")
parser.add_argument("--opt", choices=["adam", "sgd", "rmsprop"], default="sgd",
        help="Which optimizer to use")
parser.add_argument("--bce", action="store_true", 
        help="if specified, use BCE instead of l2 for reconstruction loss")
parser.add_argument("--lse", action="store_true",
        help="if specified, use log-sum-exp to control max instead of avg")
parser.add_argument("--iso", action="store_true",
        help="isometry loss for at")
parser.add_argument("-r", "--resume", type=str)
args = parser.parse_args()

BATCH_SIZE = args.batch_size
IMAGE_DIM = 32*32*3 if args.dataset == "cifar" else 784
IMAGE_SL = 32 if args.dataset == "cifar" else 28
NUM_EPOCHS = args.num_epochs

image_transform = transforms.Compose([
#        transforms.RandomCrop(IMAGE_SL, padding=4),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

DATASETS = {
        "cifar": datasets.CIFAR10,
        "mnist": datasets.MNIST
}
trainset = DATASETS[args.dataset](root=".data", train=True, transform=image_transform, download=True)
testset = DATASETS[args.dataset](root=".data", train=False, transform=test_transform, download=True)

trainloader = ch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = ch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

if args.dataset == "cifar":
    from models.encoders import CIFARAutoEncoder as AutoEncoder
else:
    from models.encoders import MNISTAutoEncoder as AutoEncoder

ae = AutoEncoder(num_feats=IMAGE_DIM, embed_feats=args.embed_feats, spectral_norm=False).cuda()
if args.resume is not None:
    ae.load_state_dict(ch.load(args.resume))

loss_fn = nn.CrossEntropyLoss()
parameters = [{'params': ae.parameters()}]

if args.opt == "sgd":
    opt = optim.SGD(ae.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200,300], gamma=args.lr_decay)
elif args.opt == "adam":
    opt = optim.Adam(ae.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[1e6])

for j in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(trainloader):
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        opt.zero_grad()
        images, labels = images.cuda(), labels.cuda()
        rec_images = ae(images.clone())
        if args.bce:
            rec_loss = F.binary_cross_entropy(rec_images, images)
        else:
            rec_loss =  ch.norm(images - rec_images).pow(2)/images.shape[0]
        loss_str = ""
        loss = rec_loss
        if args.at is not None:
            attack_ims = norm_sep_attack(ae, images, args.at_eps/8, args.at_eps, args.at_ns)
            _diff = ae(attack_ims) - ae(images)
            if args.lse:
                diff = ch.norm(_diff, dim=1).pow(2).exp().sum().log()
            elif args.iso:
                diff = (ch.norm(_diff, dim=1) - args.at_eps).pow(2).mean()
            else:
                diff = ch.norm(_diff).pow(2).sum()/images.shape[0]
            max_diff = ch.norm(_diff, dim=1).max()
            loss = loss + args.at * diff
            loss_str += "AT Loss: {at} | AT Max Diff: {md} | ".format(at=diff, md=max_diff)
        if args.at_dec is not None:
            attack_encs = decoder_norm_sep_attack(ae, images, args.at_eps/8, args.at_eps, args.at_ns)
            _diff = ae.decode(attack_encs) - ae(images)
            _diff = _diff.view(_diff.shape[0], -1)
            if args.iso:
                diff = (ch.norm(_diff, dim=1) - args.at_eps).pow(2).mean()
            else:
                diff = ch.norm(_diff).pow(2)/images.shape[0]
            max_diff = ch.norm(_diff, dim=1).max()
            loss = loss + args.at_dec * diff
            loss_str += "Decoder AT Loss: {at} | Decoder AT Max Diff: {md} | ".format(at=diff, md=max_diff)
        if args.vae_loss:
            # The "mu" component of VAE KL-divergence loss
            vae_loss = ch.sum(ae(images, no_decode=True).pow(2))/images.shape[0]
            loss_str += "VAE Loss: {vae} | ".format(vae=vae_loss)
            loss = loss + vae_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_str += "Reconstruction Loss {l} ".format(l=rec_loss)
        if i % 100 == 0:
            print("===== Training | Epoch {0} | Iteration {1} ======".format(j, i)) 
            print(loss_str)

    ae.eval()
    with ch.no_grad():
        num_correct = 0
        num_total = 0
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            rec_images = ae(images)
            loss = ch.norm(images - rec_images).pow(2)/images.shape[0]
        print("###### EPOCH COMPLETE ######")
        print("Test Loss: %f" % loss.cpu().item())
        print("############################")
    ae.train()
    scheduler.step()
    
    if j % args.save_epochs == 0:
        json.dump(vars(args), open(args.output_name + "_args.json", "w"))
        ch.save(ae.state_dict(), args.output_name + "_ae")
        orig_images = images[0].detach().cpu().numpy().transpose([1,2,0]).squeeze()
        recon_images = ae(images)[0].detach().cpu().numpy().transpose([1,2,0]).squeeze()
        imsave(args.output_name + "_orig.png", orig_images)
        imsave(args.output_name + "_recon.png", recon_images)
