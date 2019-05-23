import torch as ch
from optim import OAdam
from YellowFin_Pytorch.tuner_utils.yellowfin import YFOptimizer
import numpy as np

loss_fn = ch.nn.CrossEntropyLoss()

# redefine CrossEntropyLoss so that it is differentiable
# wrt both arguments
def ce(x,y, reduction='mean'):
    softmax = ch.nn.Softmax(dim=-1)
    logsoftmax = ch.nn.LogSoftmax(dim=-1)

    ce = -1.*ch.sum(softmax(x)*logsoftmax(y),dim=-1) 
    if reduction=='mean':
        return ch.mean(ce)
    elif reduction=='sum':
        return ch.sum(ce)
    elif reduction==None:
        return ch.sum(ce)


def normed(t, new_shape):
    return t/ch.norm(t.view(t.shape[0], -1), dim=-1).view(new_shape)

def l2_projection(new_images, orig_images, eps):
    batch_size = new_images.shape[0]
    new_images_flat = new_images.view(batch_size, -1)
    orig_images_flat = orig_images.view(batch_size, -1)
    diff = new_images_flat - orig_images_flat
    diff_norms = ch.norm(diff, dim=-1, keepdim=True) + 1e-8
    clip_mask = (diff_norms <= eps).float()
    clipped_diffs = diff*clip_mask + eps * diff/diff_norms * (1-clip_mask)
    clipped_ims = orig_images_flat + clipped_diffs
    return clipped_ims.view(orig_images.shape)

def linf_projection(new_images, orig_images, eps):
    return orig_images + ch.clamp(new_images - orig_images, -eps, eps)

def decoder_norm_sep_attack(ae, images, lr, eps, num_steps=10):
    orig_encs = ae.encode(images.clone()).detach()
    new_shape = [-1, 1]
    attack_encs = ae.encode(images.clone()).detach()
    g = ch.randn_like(attack_encs)
    g = g/ch.norm(g.view(g.shape[0], -1), dim=1).view(*new_shape)
    attack_encs = attack_encs + lr * g
    for _ in range(num_steps):
        attack_encs.requires_grad = True
        diff = ae.decode(attack_encs) - ae.decode(orig_encs)
        diff = diff.view(diff.shape[0], -1)
        loss = ch.norm(diff, dim=1).pow(2).mean()
        g, = ch.autograd.grad(loss, attack_encs)
        g = g/ch.norm(g.view(g.shape[0], -1), dim=1).view(*new_shape)
        attack_encs = attack_encs + lr * g
        attack_encs = l2_projection(attack_encs, orig_encs, eps)
        attack_encs = attack_encs.detach()
    return attack_encs

def norm_sep_attack(ae, images, lr, eps, num_steps=10):
    orig_ims = images.clone().detach()
    new_shape = [-1, 1, 1, 1]
    attack_ims = images.clone().detach()
    attack_ims.requires_grad = True
    g = ch.randn_like(attack_ims)
    g = g/ch.norm(g.view(g.shape[0], -1), dim=1).view(*new_shape)
    attack_ims = attack_ims + lr * g
    attack_ims = attack_ims.detach()
    for _ in range(num_steps):
        attack_ims.requires_grad = True
        diff = ae(attack_ims) - ae(orig_ims)
        loss = ch.norm(diff, dim=1).pow(2).mean()
        g, = ch.autograd.grad(loss, attack_ims)
        g = g/ch.norm(g.view(g.shape[0], -1), dim=1).view(*new_shape)
        attack_ims = attack_ims + lr * g
        attack_ims = l2_projection(attack_ims, orig_ims, eps)
        attack_ims = attack_ims.detach()
    return attack_ims

# Generate PGD examples
def pgd_generic(new_net, encode, orig_ims, correct_class, num_steps, lr, eps, use_noise, mode, flat=False):
    new_shape = [-1, 1] if flat else [-1, 1, 1, 1]
    projections = {
        "l2": l2_projection,
        "linf": linf_projection
    }
    proj = projections[mode]
    attack_ims = orig_ims.clone().detach()
    attack_ims.requires_grad = True
    if use_noise:
        if mode == 'linf':
            attack_ims = attack_ims + (ch.rand_like(orig_ims)*2 - 1) * eps
        else:
            scale = ch.rand(size=()) * eps
            noise = ch.randn_like(orig_ims)
            attack_ims = attack_ims + normed(noise, new_shape) * scale.cuda()
    for _ in range(num_steps):
        loss = loss_fn(new_net(encode(attack_ims.clone(), only_decode=flat)), correct_class)
        g, = ch.autograd.grad(loss, attack_ims)
        orig_shape = g.shape
        if mode == 'linf':
            g = ch.sign(g)
        elif mode == 'l2':
            g = g/(ch.norm(g.view(g.shape[0], -1), dim=1, keepdim=True).view(*new_shape))
        attack_ims = attack_ims + lr * g
        attack_ims = proj(attack_ims, orig_ims, eps)
        if not flat:
            attack_ims = ch.clamp(attack_ims, 0, 1)
    return attack_ims

# Generate PGD examples
def pgd_linf(*args):
    return pgd_generic(*args, mode="linf")

# Generate PGD examples
def pgd_l2(*args):
    return pgd_generic(*args, mode="l2")

def pgd_l2_flat(*args):
    return pgd_generic(*args, mode="l2", flat=True)

def pgd_linf_flat(*args):
    return pgd_generic(*args, mode="linf", flat=True)

def opmaxmin(cla, gan, eps, im_size=784, embed_feats=256, num_images=50, z_lr=5e-3, lambda_lr=1e-4,num_steps=1000, batch_num=None, ind=None):
    
    softmax = ch.nn.Softmax()
    logsoftmax = ch.nn.LogSoftmax()
    
    BATCH_SIZE = 1

    batch1 = ch.zeros((num_images, 1,28,28)).cuda()
    batch2 = ch.zeros((num_images, 1,28,28)).cuda()
    is_valid = ch.zeros(num_images).cuda()
    count = 0
    EPS = eps
    for i in range(num_images//BATCH_SIZE):

        z1 = ch.Tensor(ch.rand(BATCH_SIZE,embed_feats)).cuda() 
        z1.requires_grad = True
        z2 = ch.Tensor(ch.rand(z1.shape)).cuda()
        z2.requires_grad_()
        
        ones = ch.ones(z1.shape[0]).cuda()
        
        lambda_ = 1e0*ch.ones(z1.shape[0],1).cuda()
        lambda_.requires_grad = True

        opt1 = YFOptimizer([{'params':z1},{'params':z2}], lr=z_lr, clip_thresh=None, adapt_clip=False)
        opt2 = YFOptimizer([{'params':lambda_}], lr=lambda_lr, clip_thresh=None, adapt_clip=False)

        for j in range(num_steps):
                
            x1 = gan(z1)
            x2 = gan(z2)
            distance_mat = ch.norm((x1-x2).view(x1.shape[0],-1),dim=-1,keepdim=False) - EPS*ones
            
            cla_res1 = cla(x1).argmax(dim=-1)
            cla_res2 = cla(x2).argmax(dim=-1)
            
            #print('Cross entropy:%f \t distance=%f \t lambda=%f'%(ce(cla(x1),cla(x2)),distance_mat,lambda_))

            is_adv = 1 - (cla_res1==cla_res2).float()
            is_feasible = (distance_mat<=0).float() 
            not_valid = 1- (is_adv*is_feasible)
            if ch.sum(is_adv*is_feasible) == BATCH_SIZE:
#                 ind = (ch.abs(cla_res1 - cla_res2)*is_valid*is_feasible_mat).argmax(0)
                batch1[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x1
                batch2[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x2
                is_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = 1.
                break
            
            opt1.zero_grad()
            loss1 = (-1.* ch.sum(ce(cla(gan(z1)),cla(gan(z2)),reduction=None)*not_valid) + \
                     ch.sum(lambda_ * distance_mat*not_valid) + 1e-4*ch.sum(ch.norm(z1,dim=-1)*not_valid) +\
                     1e-4*ch.sum(ch.norm(z2,dim=-1)*not_valid))/ch.sum(not_valid)
            
            
            loss1.backward(retain_graph=True)
            opt1.step()
            
            for k in range(1):
                opt2.zero_grad()
                loss2 = -1.*ch.mean(lambda_ * distance_mat*(not_valid)) 
                loss2.backward()
                opt2.step()
                #lambda_ = lambda_.clamp(1e-3,1e5)
        batch1[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x1
        batch2[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = x2
        is_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = is_adv * is_feasible
    
    count = ch.sum(is_valid)
    print('number of adversarial pairs found:%d\n'%(count))

    return batch1.detach(), batch2.detach(), is_valid


'''
def attack(cla, gan, ims, labels, num_steps=L, batch_num=None, ind=None):
    ims = ims.view(-1, IMAGE_DIM)
    assert os.path.isfile("encodings/saved_latents_%d" % (batch_num,))
    zh = ch.load("encodings/saved_latents_%d" % (batch_num,))[ind*BATCH_SIZE:(ind+1)*BATCH_SIZE,...]

    zhat = zh.repeat(R, 1)
    targets = ims.repeat(R, 1)
    zhat.requires_grad_()

    not_dones_mask = ch.ones(zhat.shape[0])
    LAM = 1000*ch.ones_like(not_dones_mask)
    LAM.requires_grad_()

    opt = optim.Adam([zhat], lr=ADAM_LR)
    lam_opt = optim.SGD([LAM], lr=10000.0)

    lr_maker = StepLR(opt, step_size=I)

    for i in range(num_steps):
        opt.zero_grad()

        # Image Recovery Loss
        gen = gan(zhat)
        loss_mat = ((gen - targets)**2).mean(-1)
        loss_mat = loss_mat*(loss_mat > THR/2).float() - (loss_mat <= THR/2).float()
        total_loss = loss_mat.clone()
        ttf = targets.view(R*BATCH_SIZE,1,28,28)
        gtf = gen.view(ttf.shape)
        loss_extra = 0

        # Min-max CW loss
        for j in range(SAMPLES_PER_ITER):
            r = ch.randn_like(gtf)
            norm_r = ch.sqrt(r.view(-1, IMAGE_DIM).pow(2).sum(-1)).view(-1, 1, 1, 1)
            cla_res = cla.main(gtf + ROBUSTNESS_NORM*r/norm_r)

            cla_res_second_best = cla_res.clone()
            cla_res_second_best[:,labels.repeat(R)] = -LARGE_NUM
            true_classes = cla_res_second_best.argmax(-1)
            loss_new = cla_res[RANGE,labels.repeat(R)] - cla_res[RANGE,true_classes]
            loss_extra += loss_new

        loss_extra = loss_extra/SAMPLES_PER_ITER
        total_loss = loss_extra.mean() + total_loss * LAM
        #new_loss = ch.log(ch.exp(loss_extra).sum())*LAM
        #total_loss += new_loss

        if i % 50 == 0:
            print("Iteration %d | Distance Loss %f | Adversarial Loss %f" % (i, loss_mat.mean(), loss_extra.mean()))

        cla_mat = ch.stack(loss_extra.chunk(R, 0), 0)
        distance_mat = ch.stack(loss_mat.chunk(R, 0), 0) 
        not_dones_mask = 1 - (distance_mat <= THR).float()*(cla_mat <= -1).float()
        not_dones_mask = not_dones_mask.min(dim=0)[0].repeat(R)
        not_dones_mask = not_dones_mask.view(-1, 1)

        image_mat = ch.stack(gan(zhat).chunk(R, 0), 0)
        im_range = ch.range(0,BATCH_SIZE-1).long()

        ind = (-cla_mat - LARGE_NUM*(distance_mat > THR).float()).argmax(0) # Pick argmin of cla_mat 
        loss_at_best = cla_mat[ind,im_range]
        dists_at_best = distance_mat[ind,im_range]

        if not_dones_mask.mean() < 0.1 or i == num_steps - 1:
            zh_mat = ch.stack(zhat.chunk(R, 0), 0)
            best_ims = image_mat[ind,im_range,:]
            best_zhs = zh_mat[ind,im_range,:]
            return best_ims.clone().detach(), zhat.clone().detach()
        elif i % 1 == 0:
            print("----")
            print("Norms", dists_at_best)
            print("Losses", loss_at_best)
            #print("----")
            #print("Maximum loss (of best images)", loss_at_best.max())
            #print("Mean loss (of best images)", loss_at_best.mean())
            #print("----")
            print("Success rate: ", not_dones_mask.mean())
            print("Lambda: ", LAM)

        ((total_loss*not_dones_mask).mean()/not_dones_mask.mean()).backward(retain_graph=True)
        opt.step()

        # Lambda step
        lam_opt.zero_grad()
        (-(total_loss*not_dones_mask).mean()/not_dones_mask.mean()).backward()
        lam_opt.step()
        #LAM.data = ch.max(LAM, 0)[0]

        lr_maker.step()
'''
