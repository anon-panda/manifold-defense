import torch as ch
import json
from torchvision import transforms # Image processing
from torchvision import models as m
from torchvision import datasets # Auto downloading and processing datasets
from torch import nn # Neural networks
import torch.nn.functional as F # Neural network utilities
import torch.optim as optim # Optimizers
from models import resnet
from argparse import ArgumentParser
from attacks import norm_sep_attack

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
parser.add_argument("--train-on-latent", action="store_true",
        help="Whether to reconstruct or just train on the latent space directly")
parser.add_argument("--lr-decay", default=0.2, type=float,
        help="Rate at which to decay the learning rate")
parser.add_argument("-r", "--resume", type=str, 
        help="Resume training from a saved checkpoint")
parser.add_argument("-c", "--constraint", default="spectral_norm",
        choices=["spectral_norm", "adv_train", "gradient_penalty", "none"],
        help="How to enforce the Lipschitz constraint of the encoder")
parser.add_argument("--gp", type=float, help="Gradient penalty multiplier")
parser.add_argument("--at", type=float, help="Adv penalty multiplier")
parser.add_argument("--at-eps", type=float, help="Adv penalty multiplier")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
IMAGE_DIM = 32*32*3 if args.dataset == "cifar" else 784
IMAGE_SL = 32 if args.dataset == "cifar" else 28
NUM_EPOCHS = args.num_epochs

image_transform = transforms.Compose([
        transforms.RandomCrop(IMAGE_SL, padding=4),
        transforms.RandomHorizontalFlip(),
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
    from models.simple_models import CIFARAutoEncoder as AutoEncoder
    if args.train_on_latent:
        from models.simple_models import SmallSimpleClassifier as Model
    else:
        from models.resnet import ResNet18 as Model
else:
    from models.simple_models import MNISTAutoEncoder as AutoEncoder
    if args.train_on_latent:
        from models.simple_models import SmallSimpleClassifier as Model
    else:
        from models.simple_models import MNISTClassifier as Model

ae = AutoEncoder(num_feats=IMAGE_DIM, embed_feats=args.embed_feats, 
        no_decode=args.train_on_latent, spectral_norm=(args.constraint == "spectral_norm")).cuda()
net = Model().cuda() # Make an instance of our model

if args.resume is not None:
    net.load_state_dict(ch.load(args.resume))
    ae.load_state_dict(ch.load(args.resume + "_ae"))

loss_fn = nn.CrossEntropyLoss()
parameters = [{'params': net.parameters()}, {'params': ae.parameters()}]

opt = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200,300], gamma=args.lr_decay)

for j in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(trainloader):
        # Shape of images: (BATCH_SIZE x channels x width x height)
        # Shape of labels: (BATCH_SIZE)
        opt.zero_grad()
        images, labels = images.cuda(), labels.cuda()
        rec_images = ae(images.clone())
        pred_probs = net(rec_images.clone()) # Shape: (BATCH_SIZE x 10)
        pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
        acc = (pred_classes == labels).float().mean()
        pred_loss = loss_fn(pred_probs, labels)
        loss_str = ""
        loss = pred_loss
        loss_str += "Accuracy {acc} | Pred Loss {l} | ".format(
                acc=acc.detach().cpu().numpy(), l=pred_loss.detach().cpu().numpy())
        if args.reconstruction_loss != -1:
            rec_loss = args.reconstruction_loss * ch.norm(images - rec_images).pow(2)
            loss = loss + rec_loss
            loss_str += "Recon Loss {l} | ".format(l=rec_loss.detach().cpu().numpy())
        if args.constraint == "gradient_penalty":
            g_ims = images.clone().detach()
            g_ims.requires_grad = True
            enc_images = ae.encode(g_ims)
            norm_output = ch.norm(enc_images, dim=1).mean()
            g, = ch.autograd.grad(norm_output, g_ims, create_graph=True)
            grad_norm = (1 - ch.norm(g.view(g.shape[0], -1), dim=1).mean())**2
            loss_str += "GP Loss {l} | ".format(l=grad_norm.detach().cpu().numpy())
            loss = loss + args.gp * grad_norm
        if args.at is not None:
            attack_ims = norm_sep_attack(ae, images, args.at_eps/8, args.at_eps)
            diff = ch.norm(ae(attack_ims) - ae(images)).pow(2).sum()/images.shape[0]
            loss = loss + args.at * diff
            loss_str += "AT Loss: {at} | ".format(at=diff)
        loss_str += "Total Loss {l} | ".format(l=loss.detach().cpu().numpy())
        loss.backward()
        opt.step()
        scheduler.step()
        if i % 100 == 0:
            print("===== Training | Epoch {0} | Iteration {1} ======".format(j, i)) 
            print(loss_str)

    ae.eval()
    net.eval()
    with ch.no_grad():
        num_correct = 0
        num_total = 0
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            pred_probs = net(ae(images)) # Shape: (BATCH_SIZE x 10)
            pred_classes = pred_probs.argmax(1) # Shape: (BATCH_SIZE)
            num_correct += (pred_classes == labels).float().sum()
            num_total += BATCH_SIZE
        print("###### EPOCH COMPLETE ######")
        print("Test Accuracy: %f" % (num_correct/num_total).cpu().item())
        print("############################")
    ae.train()
    net.train()
    
    if j % args.save_epochs == 0:
        json.dump(vars(args), open(args.output_name + "_args.json", "w"))
        ch.save(ae.state_dict(), args.output_name + "_ae")
        ch.save(net.state_dict(), args.output_name)

