"""
Generic class for model training.
"""
from pathlib import Path
import matplotlib as mpl
import torch.optim as optim
import json
from abc import abstractmethod
from time import time
from uuid import uuid4
from pathlib import Path
from optimizations import gpu_mixup

import numpy as np
import torchmetrics
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from torch.cuda.amp import autocast
from tqdm import tqdm

import torch as ch
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from baseline_utils import baseline_train_loader, baseline_val_loader

mpl.use('module://imgcat')

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True)
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True)
)

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    lr=Param(float, 'learning rate', default=0.5),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=24),
    lr_peak_epoch=Param(float, 'Epoch at which LR peaks', default=5.),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.),
    distributed=Param(int, 'is distributed?', default=0),
    mixup_alpha=Param(float, 'mixup alpha', default=0)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(
        int, 'The size of the final resized validation image', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('distributed').enable_if(lambda cfg: cfg['training.distributed'] == 1).params(
    world_size=Param(int, 'number gpus', default=1),
    addr=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)


class Trainer():
    @param('baselines.use_baseline')
    def __init__(self, all_params, use_baseline, gpu=0):
        self.all_params = all_params
        self.gpu = gpu
        self.model, self.scaler = self.create_model_and_scaler()
        if not use_baseline:
            self.train_loader = self.create_train_loader()
            self.val_loader = self.create_val_loader()
        else:
            self.train_loader = baseline_train_loader()
            self.val_loader = baseline_val_loader()

        self.create_optimizer(len(self.train_loader))
        self.train_accuracy = torchmetrics.Accuracy(
            compute_on_step=False).to(self.gpu)
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu)
        }
        self.uid = str(uuid4())
        self.initialize_logger()

    @abstractmethod
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        raise NotImplementedError

    @abstractmethod
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size,
                          resolution):
        raise NotImplementedError

    @abstractmethod
    def create_model(self, architecture, tta):
        raise NotImplementedError

    @param('training.lr')
    @param('training.optimizer')
    @param('training.momentum')
    @param('training.weight_decay')
    @param('training.epochs')
    @param('training.lr_peak_epoch')
    @param('training.label_smoothing')
    def create_optimizer(self, iters_per_epoch, lr, momentum, optimizer,
                         weight_decay, epochs, lr_peak_epoch, label_smoothing):
        optimizer = optimizer.lower()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=momentum, weight_decay=weight_decay)

        schedule = (np.arange(epochs * iters_per_epoch + 1) + 1) / \
            iters_per_epoch
        schedule = np.interp(schedule, [0, lr_peak_epoch, epochs], [0, 1, 0])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, schedule.__getitem__)
        self.loss = ch.nn.CrossEntropyLoss()

    @param('training.mixup_alpha')
    def train_loop(self, epoch, mixup_alpha=None):
        model = self.model
        model.train()
        losses = []

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            images = images.to(memory_format=ch.channels_last,
                               non_blocking=True)

            if mixup_alpha:
                images, targ_a, targ_b, lam = gpu_mixup(images, target, mixup_alpha)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.model(images)
                if mixup_alpha:
                    loss_a = self.loss(output, targ_a)
                    loss_b = self.loss(output, targ_b)
                    loss_train = loss_a * lam + loss_b * (1 - lam)
                else:
                    loss_train = self.loss(output, target)

                losses.append(loss_train.detach())
                self.train_accuracy(output, target)

            # Logging
            group_lrs = []
            for _, group in enumerate(self.optimizer.param_groups):
                group_lrs.append(group['lr'])

            names = ['ep', 'iter', 'shape', 'lrs']
            values = [epoch, ix, tuple(images.shape), group_lrs]
            msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
            if ix == 0 or ix == len(self.train_loader) - 1:
                print(msg)

            iterator.set_description(msg)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        accuracy = self.train_accuracy.compute().item()
        self.train_accuracy.reset()
        loss = ch.stack(losses).mean().item()
        print('Train acc: ', accuracy)
        return loss, accuracy

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()
        losses = []

        with ch.no_grad():
            for images, target in tqdm(self.val_loader):
                images = images.to(memory_format=ch.channels_last,
                                   non_blocking=True)

                with autocast():
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    loss_val = self.loss(output, target)
                    losses.append(loss_val.detach())
                    [meter(output, target)
                     for meter in self.val_meters.values()]

        stats = {
            k: meter.compute().item() for k, meter in self.val_meters.items()
        }

        [meter.reset() for meter in self.val_meters.values()]
        loss = ch.stack(losses).mean().item()
        print('Val stats', stats)
        return loss, stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        folder = (Path(folder) / str(self.uid)).absolute()
        folder.mkdir(parents=True)

        self.log_folder = folder
        self.logging_fp = str(folder / 'log')
        self.start_time = time()

        params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
        }

        with open(folder / 'params.json', 'w+') as handle:
            json.dump(params, handle)

    def log(self, content):
        cur_time = time()
        with open(self.logging_fp, 'w') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

        print(f'>>> Logging file: {self.logging_fp}')

    @param('training.epochs')
    def train(self, epochs):
        print('Started training...')
        for epoch in range(epochs):
            train_loss, train_acc = self.train_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })
        val_loss, val_stats = self.val_loop()
        self.log({
            'val_loss': val_loss,
            **val_stats
        })
