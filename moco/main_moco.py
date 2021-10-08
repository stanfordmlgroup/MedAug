#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import json
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder

import sys

import aihc_utils.storage_util as storage_util
import aihc_utils.image_transform as image_transform
import aihc_utils.custom_datasets as custom_datasets

from moco.loss import WeightedSoftmaxMoCoLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
# JBY: Decrease number of workers
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[12, 18, 24], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
# JBY: Changed to 256 to fix size of our dataset
parser.add_argument('--moco-k', default=24576, type=int,
                    help='queue size; number of negative keys (default: 32768)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2',
                             'chexpert', 'moco_v2-chexpert'],
                    help='version of data augmentation to use')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--cos-rate', default=4, type=float, metavar='CR',
                    help='Scaling factor for cos, higher the slower the decay')


# Stanford AIHC modification
parser.add_argument('--exp-folder', dest='exp_folder', type=str,
                    default='./experiments', help='Experiment folder')
parser.add_argument('--exp-name', dest='exp_name', type=str, default='exp',
                    help='Experiment name')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to train folder')
parser.add_argument('--save-epoch', dest='save_epoch', type=int, default=1,
                    help='Number of epochs per checkpoint save')
parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
                    help='use pre-trained ImageNet model')
parser.add_argument('--img-size', dest='img_size', type=int, default=320,
                    help='image size (Chexpert=320)')
parser.add_argument('--crop', dest='crop', type=int, default=320,
                    help='image crop (Chexpert=320)')
parser.add_argument('--maintain-ratio', dest='maintain_ratio',
                    default=True,
                    action='store_true',
                    help='whether to maintain aspect ratio or scale the image')
parser.add_argument('--rotate', dest='rotate', type=int, default=10,
                    help='degree to rotate image')
parser.add_argument('--optimizer', dest='optimizer', default='adam',
                    help='optimizer to use, chexpert=adam, moco=sgd')
parser.add_argument('-sl', '--scale_lower', dest='moco_scale_lower', type=float, default=0.2,
                    help='moco random resized crop scale lower bound')
parser.add_argument('-su', '--scale_upper', dest='moco_scale_upper', type=float, default=1.0,
                    help='moco random resized crop scale upper bound')
parser.add_argument('--gray-prob', dest='grayscale_prob', type=float, default=0.2,
                    help='moco probability to run grayscale')

# AIHC - conaug modification
parser.add_argument('--same-patient', action='store_true',
                    help='images from same patient classified as positive')
parser.add_argument('--same-study', action='store_true',
                    help='images from same patient, same study classified as positive')
parser.add_argument('--diff-study', action='store_true',
                    help='images from same patient, different study classified as positive')
parser.add_argument('--same-laterality', action='store_true',
                    help='images from same patient, same laterality classified as positive')
parser.add_argument('--diff-laterality', action='store_true',
                    help='images from same patient, different laterality classified as positive')
parser.add_argument('--same-disease', action='store_true',
                    help='CHEAT: images from same patient, same disease classified as positive')
parser.add_argument('--hard-negative', default='random', choices=['random', 'lateral'],
                    help='define methods for negative pairs. "lateral"' +
                    ' means use same laterality for hard negative pairs' +
                    ' "view" means use AP/PA for hard negative pairs')
parser.add_argument('--target-hard-negative', type=float, default=None,
                    help='target percentage of hard negative pairs for' +
                    ' reweighting, typically 0.9')
parser.add_argument('--subsample', action='store_true',
                    help='subsample the front/front pairs to match the' +
                    ' the size of lateral/lateral pairs hard negatives')
parser.add_argument('--append-hard-negative', type=float, default=None,
                    help='append hard negatives to the existing queue' +
                    ' e.g. 0.1')
parser.add_argument('--synthesize-hard-samples', action='store_true',
                    help='generate hard samples to add to the existing queue' +
                    ' Need to set a value for "--append-hard-negative"')
parser.add_argument('--negative_laterality_only', action='store_true',
                    help='use same laterality only for hard negatives' +
                    ' the effective queue size is therefore shorten')


def main():
    args = parser.parse_args()

    print(args)

    exp_type = "-".join([args.arch, args.aug_setting, str(args.lr)])
    if args.from_imagenet:
        exp_type += "-pretrained"
    checkpoint_folder = storage_util.get_storage_folder(
        args.exp_folder, args.exp_name, exp_type)
    args_path = checkpoint_folder / "args.json"

    save_arguments(args, args_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(
            ngpus_per_node, args, checkpoint_folder))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, checkpoint_folder)


def main_worker(gpu, ngpus_per_node, args, checkpoint_folder):
    args.gpu = gpu

    # import pdb; pdb.set_trace()
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print(f"=> creating model '{args.arch}', pretrained={args.from_imagenet}")
    args.positive_strat = {"same_study": args.same_study,
                           "diff_study": args.diff_study,
                           "same_lat": args.same_laterality,
                           "diff_lat": args.diff_laterality}

    args.negative_strat = {"hard_neg": args.hard_negative,
                           "target_hard_neg": args.target_hard_negative,
                           "subsample": args.subsample,
                           "append_neg": args.append_hard_negative,
                           "syn_hard": args.synthesize_hard_samples,
                           'neg_lat_only': args.negative_laterality_only}

    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,
        args.from_imagenet, args.positive_strat, args.negative_strat)
    print(model)

    start_distributed = time.time()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    print('Distributed model defined, time: {}'.format(
        time.time() - start_distributed))

    # define loss function (criterion) and optimizer
    if args.target_hard_negative:
        criterion = WeightedSoftmaxMoCoLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    print('Loss defined')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)

    print('Optimizer defined')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traincsv = args.train_data

    start_augmentation = time.time()

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # Use ChexPert mean and std.
    normalize = transforms.Normalize(
        mean=[.5020, .5020, .5020], std=[.085585, .085585, .085585])

    if args.aug_setting.startswith('moco_v2'):
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(args.crop, scale=(
                args.moco_scale_lower, args.moco_scale_upper)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=args.grayscale_prob),
            transforms.RandomApply(
                [moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        # Combine chexpert with moco v2 augmentation.
        if args.aug_setting == 'moco_v2-chexpert':
            # Drop ToTensor and normalize.
            chexpert_augmentation = image_transform.get_transform(args, training=True)[
                :-2]
            augmentation = chexpert_augmentation + augmentation

    elif args.aug_setting == 'moco_v1':
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(args.crop, scale=(
                args.moco_scale_lower, args.moco_scale_upper)),
            transforms.RandomGrayscale(p=args.grayscale_prob),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.aug_setting == 'chexpert':
        augmentation = image_transform.get_transform(args, training=True)

    print('Augmentation: {}'.format(augmentation))

    start_train_data = time.time()

    train_dataset = custom_datasets.PatientPositivePairDataset(
        traincsv, transforms.Compose(augmentation), args.same_patient,
        args.same_study, args.diff_study, args.same_laterality,
        args.diff_laterality, args.same_disease)

    print('Load train dataset takes: {}'.format(time.time() - start_train_data))

    start_data_loader = time.time()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        # JBY: this should work because we only use 1 GPU
        # train_sampler = None
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print('Training dataloader defined, time: {}'.format(
        time.time() - start_data_loader))

    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion,
              optimizer, epoch, args, checkpoint_folder)

        if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0 and
             ((epoch % args.save_epoch == 0) or (epoch == args.epochs - 1))):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(
                checkpoint_folder, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
        print("Epoch: {} finished in time: {}".format(
            epoch, time.time() - start_epoch))
        print("=" * 50)


def train(train_loader, model, criterion, optimizer, epoch, args,
          checkpoint_folder=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        checkpoint_folder=checkpoint_folder)

    # switch to train mode
    model.train()

    print(f'Running epoch {epoch}')

    end = time.time()
    for i, (images, meta_info, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        # compute output
        output, target, weight = model(im_q=images[0], im_k=images[1],
                                       meta_info=meta_info)
        if args.target_hard_negative:
            loss = criterion(output, weight)
        else:
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        i += 1

        sys.stdout.flush()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_arguments(args, path):
    args_dict = vars(copy.deepcopy(args))
    with open(path, 'w') as args_file:
        json.dump(args_dict, args_file, indent=4, sort_keys=True)
        args_file.write('\n')
    print('Arguments saved to: {}'.format(path))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", checkpoint_folder=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.display_and_log = False
        if checkpoint_folder is not None:
            self.display_and_log = True
            self.log_file = open(os.path.join(
                checkpoint_folder, "train_log.txt"), "w")
            print('Train history logged to: {}'.format(self.log_file))

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if self.display_and_log:
            # print('\t'.join(entries) + '\n', file=self.log_file)
            self.log_file.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        # TODO, JBY, is /4 an appropriate scale?
        lr *= 0.5 * (1. + math.cos(math.pi * epoch /
                                   args.epochs / args.cos_rate))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
