from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
from sklearn.neighbors import KDTree
from torch_kdtree import build_kd_tree
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.coal_label import *
import models.coal as Models

class DisMeasure:
    def __init__(self, sourceClouds):
        self.sourceClouds = sourceClouds.to(torch.float32)

    def match_by_hausdorffdis(self, inCloud):
        
        tree = build_kd_tree(inCloud)
        
        mindis = 999
        for i, sourceCloud in enumerate(self.sourceClouds):
            hausdis = 0
            dists, _ = tree.query(sourceCloud, nr_nns_searches=1)
            hausdis = max(dists)
    
            if mindis > hausdis:
                mindis = hausdis
                minindex = i
    
        return mindis, minindex
        

def save_ply(points, colors, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % points.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            f.write('%f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], colors[i][0], colors[i][1], colors[i][2]))

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, disMeasure, train_p4t, train_tempalteNet, templates):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))

    cor = 0    
    total = 0

    header = 'Epoch: [{}]'.format(epoch)
    for pc1, rgb1, label1, template_vals, index in metric_logger.log_every(data_loader, print_freq, header):
        
        template_vals = template_vals.to(device)
        template_vals = template_vals.squeeze()
        
        _, template_pres = torch.min(template_vals, dim=1)
        # template_vals = []
        # for clip in pc1:
        #     clip_pre = []
        #     for pc in clip:
        #         template_pre = disMeasure.match_by_hausdorffdis(pc.to(device))
        #         clip_pre.append(template_pre)
        #     template_vals.append(clip_pre)
        # template_vals = torch.tensor(template_vals).to(device)
        criterion_template = nn.CrossEntropyLoss(weight=torch.ones(8).to(device), reduction='none')
        mse_template = nn.MSELoss()
            
        
        start_time = time.time()

        pc1, rgb1, label1, templates = pc1.to(device), rgb1.to(device), label1.to(device), templates.to(device).to(torch.float32)
        output1, template_out, connect_out = model(pc1, rgb1, templates)
        output1 = output1.transpose(1, 2)
        connect_out = connect_out.transpose(1, 2)
        
        # choose output
        out = output1
        loss1 = criterion(out, label1)
        
        # template_loss = criterion_template(template_out, template_vals)
        template_loss = mse_template(template_vals, template_out)
        
        _, template_choice = torch.min(template_out, dim=1)
        
        cor = cor + torch.sum(template_choice == template_pres)
        total = total + 24
        
        weight = (-label1 + 1) * 79.0 + 1 # 40:1
        loss1 = torch.sum(loss1 * weight) / (label1.shape[0] * label1.shape[1] * label1.shape[2])
        # template_loss = torch.sum(template_loss) / (template_out.shape[0] * template_out.shape[1])
        optimizer.zero_grad() 
        if(train_p4t):
            loss1.backward(retain_graph=True)
        if(train_tempalteNet):
            template_loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss1.item(), lr=optimizer.param_groups[0]["lr"])
        lr_scheduler.step()
        sys.stdout.flush()
    
    acc = cor / total
    print("acc: {}".format(acc))

def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = CoalLabelDataset(
            root=args.data_path,
            file=args.data_file,
            meta=args.data_train,
            frames_per_clip=args.clip_len,
            num_points=args.num_points,
            train=True
    )
    
    templates = np.load(os.path.join(args.data_path, 'coal_template.npz'))['pc']
    templates = torch.tensor(templates)
    disMeasure = DisMeasure(templates)

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    
    Model = getattr(Models, args.model)
    model = Model(radius=args.radius, nsamples=args.nsamples, num_classes=2)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion_train = nn.CrossEntropyLoss(weight=torch.from_numpy(dataset.labelweights).to(device), reduction='none')

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion_train, optimizer, lr_scheduler, data_loader, device, epoch, args.print_freq, disMeasure, args.train_p4t, args.train_templateNet, templates)

        if args.output_dir:
            output_dir = args.output_dir
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Model Training')

    parser.add_argument('--data-path', default='/scratch/HeheFan-data/Synthia4D/sequences', help='data path')
    parser.add_argument('--data-file', default='coal.npz')
    parser.add_argument('--data-train', default='/scratch/HeheFan-data/Synthia4D/trainval_raw.txt', help='meta list for training')
    parser.add_argument('--data-eval', default='/scratch/HeheFan-data/Synthia4D/test_raw.txt', help='meta list for test')
    parser.add_argument('--label-weight', default='/scratch/HeheFan-data/Synthia4D/labelweights.npz', help='training label weights')
    
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='P4Transformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=3, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num-points', default=16384, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.9, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=16, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=1, type=int, help='temporal kernel size')
    # embedding
    parser.add_argument('--emb-relu', default=False, action='store_true')
    # transformer
    parser.add_argument('--dim', default=1024, type=int, help='transformer dim')
    parser.add_argument('--depth', default=2, type=int, help='transformer depth')
    parser.add_argument('--head', default=4, type=int, help='transformer head')
    parser.add_argument('--mlp-dim', default=2048, type=int, help='transformer mlp dim')
    # training
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[30, 40, 50], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # training type
    parser.add_argument('--train-p4t', default=True, action='store_true', help='enable p4t training')
    parser.add_argument('--train-templateNet', default=True, action='store_false', help='enable templateNet training')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
