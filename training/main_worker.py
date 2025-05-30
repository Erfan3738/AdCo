import builtins
import torch.distributed as dist
import os
#import torchvision.models as models
import model.ResNet as models
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime
import time
import numpy as np

from model.AdCo import AdCo, Adversary_Negatives
from ops.os_operation import mkdir
from training.train_utils import adjust_learning_rate,save_checkpoint
from training.train import train, init_memory
from data_processing.loader import TwoCropsTransform, GaussianBlur
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from ops.knn import knn_monitor

def init_log_path(args):
    """
    :param args:
    :return:
    save model+log path
    """
    save_path = os.path.join(os.getcwd(), args.log_path)
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, args.dataset)
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_memlr" + str(args.memory_lr))
    if args.gpu == 0:
        mkdir(save_path)
    save_path = os.path.join(save_path, "cos_" + str(args.cos))
    if args.gpu == 0:
        mkdir(save_path)
    import datetime
    today = datetime.date.today()
    formatted_today = today.strftime('%y%m%d')
    now = time.strftime("%H:%M:%S")
    save_path = os.path.join(save_path, formatted_today + now)
    if args.gpu == 0:
        mkdir(save_path)
    return save_path

def main_worker(gpu, args):
    params = vars(args)
    
    print(vars(args))



    print("=> creating model '{}'".format(args.arch))
    multi_crop = args.multi_crop
    Memory_Bank = Adversary_Negatives(args.cluster,args.moco_dim, multi_crop)

    model = AdCo(models.__dict__[args.arch], args,
                           args.moco_dim, args.moco_m, args.moco_t, args.mlp)
#     print(model)


    model.cuda()
    Memory_Bank.cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    from model.optim import  LARS
    #optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    #optimizer = LARS(model.parameters(), args.lr ,weight_decay=args.weight_decay,momentum=args.momentum)

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
            Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset=='ImageNet':
        traindir = os.path.join(args.data, 'train')
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        if args.multi_crop:
            from data_processing.MultiCrop_Transform import Multi_Transform
            multi_transform = Multi_Transform(args.size_crops,
                                              args.nmb_crops,
                                              args.min_scale_crops,
                                              args.max_scale_crops, normalize)
            train_dataset = datasets.ImageFolder(
                traindir, multi_transform)
        else:
            augmentation = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            train_dataset = CIFAR10(root='./datasets', train=True, download=True, transform=TwoCropsTransform(augmentation))

    else:
        print("We only support ImageNet dataset currently")
        exit()
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    val_dataset =CIFAR10(root='./datasets', train=True, download=True, transform=transform_test)
    test_dataset =CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
            
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,pin_memory=True,num_workers=args.workers,drop_last=False)
            

    save_path = init_log_path(args)
    #init weight for memory bank
    bank_size=args.cluster

    model.eval()
    #init memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        init_memory(train_loader, model, Memory_Bank, criterion,
              optimizer, 0, args)
        print("Init memory bank finished!!")

    best_Acc=0
    knn_path = os.path.join(save_path,"knn.log")
    for epoch in range(args.start_epoch, args.epochs):


        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1 = train(train_loader, model,Memory_Bank, criterion,
                                optimizer, epoch, args)
        if epoch%5 ==0 or epoch<=20:
            print("gpu consuming before cleaning:", torch.cuda.memory_allocated()/1024/1024)
            torch.cuda.empty_cache()
            print("gpu consuming after cleaning:", torch.cuda.memory_allocated()/1024/1024)
            acc=knn_monitor(model.encoder_q, val_loader, test_loader,epoch, args,global_k = 10) 
            print({'*KNN monitor Accuracy': acc})
            with open(knn_path,'a+') as file:
                file.write('%d epoch KNN monitor Accuracy %f\n'%(epoch,acc))
            
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }


            if epoch%10==9:
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')

            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
