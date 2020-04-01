#
# Obtain hyperspherical prototypes prior to network training.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from helper import Logger

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

#
# PArse user arguments.
#

parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
parser.add_argument('-c', dest="classes", default=100, type=int)
parser.add_argument('-d', dest="dims", default=100, type=int)
parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
parser.add_argument('-m', dest="momentum", default=0.9, type=float)
parser.add_argument('-e', dest="epochs", default=10000, type=int, )
parser.add_argument('-s', dest="seed", default=300, type=int)
parser.add_argument('-r', dest="resdir", default="", type=str)
parser.add_argument('-w', dest="wtvfile", default="", type=str)
args = parser.parse_args()

#
# Compute the loss related to the prototypes.
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t())
    # Remove diagnonal from loss
    product_ = product - 2. * torch.diag(torch.diag(product))
    # Compute the min and max theta
    min_theta = torch.acos(product_.max())
    max_theta = torch.acos(product.min())

    if args.losstype == 'hpn':
        # Minimize maximum cosine similarity.
        loss = product_.max(dim=1)[0].mean()

    elif args.losstype == 'hpn-theta':
        # Maxmize min theta.
        theta = torch.acos(product_.max(dim=1)[0].clamp(-0.99999, 0.99999))
        loss = -theta.mean()

    elif args.losstype in ['hpn-global-cosine', 'hpn-global-theta']:
        triu = []
        for i in range(product.size(0)-1):
            triu.append(product[i, i + 1:])
        cosine = torch.cat(triu).clamp(-0.9999, 0.9999)

        if args.losstype == 'hpn-global-cosine':
            loss = (cosine + 1.0).pow(2).mean()
        else:
            loss = (torch.acos(cosine) - np.pi).pow(2).mean()

    elif args.losstype in ['s-kernel', 'Lennard-Jones', 'Lennard-Jones-MSE']:
        diff=[]
        for i in range(prototypes.size(0)-1):
            diff.append(prototypes[i] - prototypes[i+1:])
        norm = torch.cat(diff).norm(p=2, dim=1)

        if args.losstype == 's-kernel':
            if args.loss_skernel_s > 0:
                loss = norm.pow(-args.loss_skernel_s).mean()
            elif args.loss_skernel_s == 0:
                loss = -torch.log(norm).mean()
        else:
            norm = args.loss_lj_r0/norm
            if args.losstype == 'Lennard-Jones':
                loss = (norm.pow(2*args.loss_lj_p) - norm.pow(args.loss_lj_p)).mean()
            else:
                loss = torch.abs(norm.pow(2*args.loss_lj_p) - norm.pow(args.loss_lj_p)).mean()

    return loss, min_theta, max_theta

#
# Compute the semantic relation loss.
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:,0], triplets[:,1]]
    loss2 = product[triplets[:,2], triplets[:,3]]
    return loss1.mean() + loss2.mean(), product.max()

#
# Main entry point of the script.
#
def perform():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")
    # kwargs = {'num_workers': 64, 'pin_memory': True}

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # global tb_summary_writer
    # tb_summary_writer = tb and tb.SummaryWriter(args.resdir)

    # if not os.path.isdir(args.resdir):
    #     os.makedirs(args.resdir)

    # Initialize prototypes and optimizer.
    if os.path.exists(args.wtvfile):
        use_wtv = True
        wtvv = np.load(args.wtvfile)
        for i in range(wtvv.shape[0]):
            wtvv[i] /= np.linalg.norm(wtvv[i])
        wtvv = torch.from_numpy(wtvv)
        wtvsim = torch.matmul(wtvv, wtvv.t()).float()
        
        # Precompute triplets.
        nns, others = [], []
        for i in range(wtvv.shape[0]):
            sorder = np.argsort(wtvsim[i,:])[::-1]
            nns.append(sorder[:args.nn])
            others.append(sorder[args.nn:-1])
        triplets = []
        for i in range(wtvv.shape[0]):
            for j in range(len(nns[i])):
                for k in range(len(others[i])):
                    triplets.append([i,j,i,k])
        triplets = np.array(triplets).astype(int)
    else:
        use_wtv = False
    
    # Initialize prototypes.
    prototypes = torch.randn(args.classes, args.dims)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))

    optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
            momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=100)

    # Optimize for separation.
    min_loss = 0.0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        if lr < 1e-6:
            break
        # Compute loss
        normalized_proto = F.normalize(prototypes, p=2, dim=1)
        loss1, min_theta, max_theta = prototype_loss(normalized_proto)
        if use_wtv:
            loss2 = prototype_loss_sem(prototypes, triplets)
            loss = loss1 + loss2
        else:
            loss = loss1
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(loss1)

        print('Epoch: [%d | %d] LR: %f  Loss: %.4f  Min_theta: %.5f  Max_theta: %.5f' % (
            epoch + 1, args.epochs, lr, loss, min_theta / np.pi * 180, max_theta / np.pi * 180))
        sys.stdout.flush()

        # tb_summary_writer.add_scalar('learning_rate', lr, epoch + 1)
        #
        if epoch == 0:
            min_loss = loss

        # Store result.
        if loss < min_loss:
            np.save(args.resdir + "/%dd-%dc.npy" % (args.dims, args.classes),
                    F.normalize(prototypes, p=2, dim=1).data.numpy())

    np.save(args.resdir + "/final-%dd-%dc.npy" % (args.dims, args.classes),
            F.normalize(prototypes, p=2, dim=1).data.numpy())


if __name__ == "__main__":

    # local config
    args.classes = 100
    args.dims = 1000
    args.learning_rate = 0.1
    args.momentum = 0.9
    args.epochs = 10000

    args.losstype = 'hpn-theta'
    args.loss_skernel_s = 1  # 0, 1, 2
    args.loss_lj_r0 = 1.2
    args.loss_lj_p = 6


    # for r in np.arange(0.1, 3.1, 0.1):
    #     for p in np.arange(1, 7, 1):
    #         args.loss_lj_r0 = r
    #         args.loss_lj_p = p
    #
    #         print('\nR: %.1f, 　P: %d\r'% (args.loss_lj_r0, args.loss_lj_p))

    args.resdir = 'prototypes/used/loss-hpn'
    if not os.path.isdir(args.resdir):
        os.makedirs(args.resdir)
    sys.stdout = Logger(args.resdir + '/log.txt')

    perform()

    # local config