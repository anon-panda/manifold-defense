import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from models.simple_models import SmallSimpleClassifier
from torchvision import transforms
from attacks import pgd_l2, pgd_linf, pgd_l2_flat, opmaxmin, ce
from argparse import ArgumentParser
from models import resnet
import numpy as np
from YellowFin_Pytorch.tuner_utils.yellowfin import YFOptimizer
import time

parser = ArgumentParser()
parser.add_argument('--num-pgd', type=int, default=10,
        help="Number of steps to use while making adversarial examples for training")
parser.add_argument('--mode', type=str, choices=['linf', 'l2', 'nat'], 
        help="Number of examples to generate")
parser.add_argument('--eps', type=float, 
        help="Adversarial perturbation budget")
parser.add_argument('--pgd-lr', type=float, 
        help="Learning rate for PGD")
parser.add_argument('--net', type=str, required=True,
        help="Path to encoder network")
parser.add_argument('--op-generator', type=str,
        help="Path to generator network for overpowered attack")
parser.add_argument('--attack-latent', action='store_true',
        help="Adversarial attack in the latent space")
parser.add_argument('--trainable-encode', action='store_true',
        help="Should allow encode layer to be trainable")
parser.add_argument('--trainable-decode', action='store_true',
        help="Should allow decode layer to be trainable")
parser.add_argument('--use-orig', action='store_true',
        help="Should use original network (ResNet) instead of simple one")
parser.add_argument('--no-norm', action='store_true',
        help="Should not normalize the image to the proper imagenet stats")
parser.add_argument('--from-scratch', action='store_true', 
        help="Should not use the pretrained embedder")
parser.add_argument('--sgd-lr', type=float, default=1e-2,
        help="SGD learning rate")
parser.add_argument('--save-str', type=str, default="",
        help="A unique identifier to save with")
parser.add_argument('--dataset', choices=["mnist", "cifar"], required=True,
        help="Which dataset to use for training")
parser.add_argument('--num-epochs', default=500, type=int,
        help="Number of epochs to train for")
parser.add_argument('--dataset-size', default=None, type=int,
        help="Number of samples to use")
parser.add_argument('--random-step', action='store_true', 
        help="Whether to start by taking a random step in PGD attack")
parser.add_argument('--no-sn', action='store_true', 
        help="If given, don't spectral normalize")
parser.add_argument('--op-attack', action='store_true', 
        help="If given, don't spectral normalize")
parser.add_argument('--width', type=int, default=10)
parser.add_argument('--with-decode', action='store_true',
        help="If given, use decoder + ConvNet/ResNet instead")
parser.add_argument('--validation-set', action='store_true',
        help="If given, use a validation set")
parser.add_argument('--opt', default="sgd", choices=["adam", "sgd", 'yf'])
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--embed-feats', type=int, default=128)
parser.add_argument('--op-embed-feats', type=int, default=20)
parser.add_argument('--resume', type=str)
args = parser.parse_args()


MODE = args.mode
MODE += "_flat" if args.attack_latent else ""
IMAGE_DIM = 32*32*3 if args.dataset == "cifar" else 784

print(args.op_attack)
if args.dataset == "cifar":
    from models.encoders import CIFARAutoEncoder as AutoEncoder
    if not args.validation_set:
        from cifar_config import no_norm_fullloader as trainloader
        from cifar_config import no_norm_testloader as testloader
    else:
        from cifar_config import no_norm_trainloader as trainloader
        from cifar_config import no_norm_testloader as testloader
        from cifar_config import no_norm_validationloader as validationloader
    from models.resnet import ResNet18 as BigModel
else:
    from models.encoders import MNISTAutoEncoder as AutoEncoder
    from models.encoders import MNISTVAE as VAE
    from models.simple_models import MNISTClassifier as BigModel
    if not args.validation_set:
        from mnist_config import testloader
        from mnist_config import trainloaderfull as trainloader
    else:
        from mnist_config import trainloader, testloader, validationloader
    
if args.baseline:
    assert args.with_decode
    from models.encoders import IdentityEncoder as AutoEncoder
    
    if args.op_attack:
        vae = VAE(IMAGE_DIM, args.op_embed_feats, spectral_norm=False).cuda()
        vae.load_state_dict(ch.load(args.op_generator))

        vae.eval()

ae = AutoEncoder(num_feats=IMAGE_DIM, embed_feats=args.embed_feats, no_decode=(not args.with_decode), \
                    spectral_norm=(not args.no_sn)).cuda()

if args.with_decode:
    net = BigModel().cuda()
else:
    net = SmallSimpleClassifier().cuda()

if not args.from_scratch and not args.baseline:
    ae.load_state_dict(ch.load(args.net + "_ae"))

if args.resume:
    ae_dict = ch.load("results/retrained_enc_%s_%s" % (MODE, args.resume))
    net_dict = ch.load("results/trained_net_%s_%s" % (MODE, args.resume))
    ae.load_state_dict(ae_dict)
    net.load_state_dict(net_dict)


net = ch.nn.DataParallel(net)
ae = ch.nn.DataParallel(ae)

NUM_STEPS = args.num_pgd
DEFAULT_EPS = {
    "l2": 0.1,
    "linf": 8/255.0,
    "nat": 2.0
}
EPS = args.eps if args.eps is not None else DEFAULT_EPS[args.mode]
LR = args.pgd_lr if args.pgd_lr is not None else 2*EPS/NUM_STEPS

NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

SAVE_ITERS = 5 
ATTACKS = {
    "l2": pgd_l2,
    "linf": pgd_linf,
    "l2_flat": pgd_l2_flat
}
attack = ATTACKS[MODE]

if args.use_orig:
    raise NotImplementedError

def encode(x, bypass=False, no_decode=False, only_decode=False):
    if only_decode:
        return ae.decode(x)
    if bypass:
        return x
    if not args.no_norm:
        x = ch.stack([NORMALIZER(x[i]) for i in range(x.shape[0])])
    if args.use_orig:
        return x
    return ae(x, no_decode=no_decode)

loss_fn = ch.nn.CrossEntropyLoss()
if args.trainable_encode:
    param_set = [{'params': net.parameters()}, {'params': ae.parameters()}]
elif args.trainable_decode:
    param_set = [{'params': net.parameters()}, {'params': ae.decode_vars}]
else:
    param_set = net.parameters()

if args.opt == "sgd":
    opt = optim.SGD(param_set, lr=args.sgd_lr, momentum=0.9, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[50,100,150,500], gamma=0.1)
elif args.opt == "adam":
    opt = optim.Adam(param_set, lr=args.sgd_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.num_epochs+1], gamma=0.1)
elif args.opt == 'yf':
    opt = YFOptimizer(param_set, lr=args.sgd_lr, clip_thresh=None, adapt_clip=False)
    #scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.num_epochs+1], gamma=0.1)


def se(x1, x2, reduction='mean'):
    y = ch.norm((x1-x2).view(x1.shape[0],-1),dim=-1,p=2)**2
    if reduction=='sum':
        return ch.sum(y)
    elif reduction=='mean':
        return ch.mean(y)
    else: 
        return y

best_adv_acc = 0.
for ep in range(1,args.num_epochs+1):
    start = time.time()
    if args.op_attack and (ep-1) % 8==0:
        net.train()
        opt.zero_grad()
        batch1, batch2, is_adv = opmaxmin(net,vae.decode,EPS,num_images=50,\
                num_steps=500,embed_feats=args.op_embed_feats,z_lr=1e-4,lambda_lr=1e-4,ind=ep)
        if ch.sum(is_adv) > 0:
            loss = 1e-2*ch.sum(ce(net(batch1), net(batch2),reduction=None)*is_adv)/ch.sum(is_adv)
            loss.backward()
            opt.step()
        else:
            pass
    
    total_ims_seen = 0
    val_num_correct = 0
    val_num_total = 0
    for i, (images, labels) in enumerate(trainloader):

        if args.dataset_size is not None and total_ims_seen > args.dataset_size:
            break
        ae.train()
        net.train()
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        images, labels = images.cuda(), labels.cuda()
        if MODE != "nat":
            if args.attack_latent:
                enc_images = encode(images.clone(), no_decode=True)
            else:
                enc_images = images
            _args = [net, encode, enc_images, labels, NUM_STEPS, LR, EPS, args.random_step]
            images = attack(*_args)
        new_ims = encode(images.clone(), only_decode=args.attack_latent)
        opt.zero_grad()

        pred_probs = net(new_ims)
        loss = loss_fn(pred_probs, labels)
        pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
        acc = (pred_classes == labels).float().mean()
        if (i+1) % 25 == 0:
            print("Epoch {0} | Iteration {1} | Loss {2} | Adv Acc {3}".format(ep, i+1, loss, acc))
        loss.backward()
        opt.step()
        total_ims_seen += images.shape[0]

    end = time.time()
    scheduler.step()
    print('time for epoch: %f seconds'%(end-start))
    
    if args.validation_set and args.op_attack :
        for i, (images, labels) in enumerate(validationloader):
            ae.eval()
            net.eval()
            # Shape of images: (BATCH_SIZE x channels x width x height)
            # Shape of labels: (BATCH_SIZE)
            images, labels = images.cuda(), labels.cuda()
            if MODE != "nat":
                if args.attack_latent:
                    enc_images = encode(images.clone(), no_decode=True)
                else:
                    enc_images = images
                _args = [net, encode, enc_images, labels, NUM_STEPS, LR, EPS, args.random_step]
                images = attack(*_args)
            new_ims = encode(images.clone(), only_decode=args.attack_latent)
            pred_probs = net(new_ims)
            loss = loss_fn(pred_probs, labels)
            pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)

            val_num_correct += (pred_classes == labels).float().sum()
            val_num_total += labels.shape[0]
        

        print("###### EPOCH {0} COMPLETE ######".format(ep))
        print("Adversarial Validation Accuracy: %f" % (val_num_correct/val_num_total).cpu().item())
        print("############################")
        if val_num_correct/val_num_total > best_adv_acc:
            ch.save(ae.state_dict(), "results/retrained_enc_%s_%s" % (MODE, args.save_str))
            ch.save(net.state_dict(), "results/trained_net_%s_%s" % (MODE, args.save_str))
            print("Saved model...")
            best_adv_acc = val_num_correct/val_num_total

    ae.eval()
    net.eval()
    with ch.no_grad():
        num_correct = 0
        num_total = 0
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            new_ims = encode(images)
            pred_probs = net(new_ims) # Shape: (BATCH_SIZE x 10)
            pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
            num_correct += (pred_classes == labels).float().sum()
            num_total += labels.shape[0]
        print("###### EPOCH {0} COMPLETE ######".format(ep))
        print("Test Accuracy: %f" % (num_correct/num_total).cpu().item())
        print("############################")
    
    num_correct = 0
    num_total = 0
    net.eval()
    for (images, labels) in testloader:
        images, labels = images.cuda(), labels.cuda()
        if args.attack_latent:
            images = encode(images, no_decode=True)
        _args = [net, encode, images, labels, NUM_STEPS, LR, EPS, False]
        images = attack(*_args)
        pred_probs = net(encode(images, only_decode=args.attack_latent)) # Shape: (BATCH_SIZE x 10)
        pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
        num_correct += (pred_classes == labels).float().sum()
        num_total += labels.shape[0]
    print("###### EPOCH {0} COMPLETE ######".format(ep))
    print("Adversarial Test Accuracy: %f" % (num_correct/num_total).cpu().item())
    print("############################")
    
    if ep % SAVE_ITERS == 0:
        ch.save(ae.module.state_dict(), "results/retrained_enc_%s_%s_%d" % (MODE, args.save_str,ep))
        ch.save(net.module.state_dict(), "results/trained_net_%s_%s_%d" % (MODE, args.save_str,ep))
        print("Saved model...")
   
