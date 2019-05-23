import torch as ch
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import sys
from models import resnet
from torchvision import transforms
from attacks import pgd_l2, pgd_linf, pgd_l2_flat
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--net", type=str, required=True,
        help="The path to the network to evaluate")
parser.add_argument("--ae", type=str, required=True,
        help="The path to the autoencoder to evaluate")
parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar"],
        help="What dataset to evaluate against")
parser.add_argument("--num-steps", type=int, default=100,
        help="Number of steps of PGD to run")
parser.add_argument("--eps", type=float, required=True,
        help="Adversarial perturbation budget")
parser.add_argument("--no-sn", action="store_true", 
        help="If given, don't use spectral norm")
parser.add_argument("--decode", action="store_true",
        help="If given, use the decoder too")
parser.add_argument('--baseline', action='store_true',
        help="If given, just use classifier")
args = parser.parse_args()

if args.decode:
    from models.simple_models import MNISTClassifier as Model
else:
    from models.simple_models import SmallSimpleClassifier as Model

if args.dataset == "cifar":
    from models.encoders import CIFARAutoEncoder as AutoEncoder
    from cifar_config import no_norm_testloader as testloader
else:
    from models.encoders import MNISTAutoEncoder as AutoEncoder
    from mnist_config import testloader

if args.baseline:
    from models.encoders import IdentityEncoder as AutoEncoder

print(not args.decode, not args.no_sn)
ae = AutoEncoder(num_feats=(32*32*3 if args.dataset=="cifar" else 784), 
        embed_feats=128, no_decode=(not args.decode), spectral_norm=(not args.no_sn)).cuda()
net = Model().cuda()

ae.load_state_dict(ch.load(args.ae))
net.load_state_dict(ch.load(args.net))

ae.eval()
net.eval()

NUM_STEPS = args.num_steps
attack = pgd_l2
EPS = args.eps
LR = 0.5
NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
SAVE_ITERS = 10
loss_fn = ch.nn.CrossEntropyLoss()

def encode(x, bypass=False, normalize=False, only_decode=False):
    if only_decode:
        return ae.decode(x)
    if bypass:
        return x
    if normalize:
        x = ch.stack([NORMALIZER(x[i]) for i in range(x.shape[0])])
    return ae(x)

num_correct = 0
num_total = 0
for (images, labels) in testloader:
    images, labels = images.cuda(), labels.cuda()
    #images = ae.encode(images)#encode(images)
    args = [net, encode, images.clone(), labels, NUM_STEPS, LR, EPS, False]
    attack_images = attack(*args)
    #new_ims = ae.decode(images.clone())
    print("----")
    print(ch.norm((attack_images - images).view(images.shape[0],-1), dim=1))
    print(ch.norm((encode(attack_images.clone()) - encode(images.clone())).view(images.shape[0], -1), dim=1))
    print("----")
    new_ims = encode(attack_images.clone())
    #print("----")
    #print(ch.norm((new_ims - encode(images.clone())).view(images.shape[0],-1), dim=1))
    #print("----")
    pred_probs = net(new_ims) # Shape: (BATCH_SIZE x 10)
    pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
    num_correct += (pred_classes == labels).float().sum()
    num_total += labels.shape[0]
    print(num_correct/num_total)
print("###### EPOCH COMPLETE ######")
print("Adversarial Accuracy: %f" % (num_correct/num_total).cpu().item())
print("############################")
