# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torchvision.models import resnet
from functools import partial

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, num_classes=128, arch=None, bn_splits=1):
        super(ModelBase, self).__init__()
        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=num_classes, norm_layer=norm_layer)
        
        self.features = []
        self.fc = None
        
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.features.append(nn.Flatten(1))
                self.fc = module
                continue
            self.features.append(module)
        
        self.features = nn.Sequential(*self.features)
        
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        # note: not normalized here
        return x


class AdCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder,args, dim=128, m=0.999, T=0.07, mlp=True, arch='resnet18', bn_splits=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdCo, self).__init__()
        self.args=args
        self.m = m
        self.T = T
        self.T_m = args.mem_t
        self.sym = args.sym
        self.multi_crop = args.multi_crop
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = ModelBase(num_classes=dim, arch=arch, bn_splits=bn_splits)
        #self.encoder_q = base_encoder(num_classes=dim)
        #self.encoder_q.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        #self.encoder_q.maxpool = nn.Identity()
        self.encoder_k = ModelBase(num_classes=dim, arch=arch, bn_splits=bn_splits)
        #self.encoder_k.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        #self.encoder_k.maxpool = nn.Identity()
        
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K=args.cluster


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
        
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle
        
    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.multi_crop:
            q_list = []
            for k, im_q in enumerate(im_q):  # weak forward
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)
                # q = self._batch_unshuffle_ddp(q, idx_unshuffle)
                q_list.append(q)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k = k.detach()
            return q_list, k
        #elif not self.sym:
           # q = self.encoder_q(im_q)  # queries: NxC
           # q = nn.functional.normalize(q, dim=1)
            # compute key features
           # with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
               # self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
               # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

               # k = self.encoder_k(im_k)  # keys: NxC
               # k = nn.functional.normalize(k, dim=1)
                
                # undo shuffle
               # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
               # k = k.detach()

           # return q, k
        else:
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            q_pred=q
            k_pred = self.encoder_q(im_k)  # queries: NxC
            k_pred = nn.functional.normalize(k_pred, dim=1)
            with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
                self._momentum_update_key_encoder()  # update the key encoder

                im_q_, idx_unshuffle = self._batch_shuffle_single_gpu(im_q)
                q = self.encoder_k(im_q_)  # keys: NxC
                q = nn.functional.normalize(q, dim=1)
                q = self._batch_unshuffle_single_gpu(q, idx_unshuffle)
                q = q.detach()


                im_k_, idx_unshuffle1 = self._batch_shuffle_single_gpu(im_k)
                k = self.encoder_k(im_k_)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle_single_gpu(k, idx_unshuffle1)
                k = k.detach()
                
            return q_pred,k_pred,q, k

class Adversary_Negatives(nn.Module):
    def __init__(self,bank_size,dim,multi_crop=0):
        super(Adversary_Negatives, self).__init__()
        self.multi_crop = multi_crop
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q, init_mem=False):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        if self.multi_crop and not init_mem:
            logit_list = []
            for q_item in q:
                logit = torch.einsum('nc,ck->nk', [q_item, memory_bank])
                logit_list.append(logit)
            return memory_bank, self.W, logit_list
        else:
            logit=torch.einsum('nc,ck->nk', [q, memory_bank])
            return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
