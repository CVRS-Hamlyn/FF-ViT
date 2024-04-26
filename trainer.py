from __future__ import absolute_import, division, print_function
from ast import Or
import imp
from math import gamma
from operator import index
from random import sample, shuffle
import time
from tkinter import ON
from turtle import clear, distance, pos, width
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import data_loader
import network
from utils import *
from tqdm.autonotebook import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as D
import time
import json
import copy
from data_loader import pCLE_Rotation_dataset, Heart_dataset
from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
import torch.backends.cudnn as cudnn
import seaborn as sns
sns.set(rc={"figure.figsize":(18, 10)}) 



class Trainer:
    def __init__(self, options):
        self.opts = options

        #   self.log_path = os.path.join(self.opts.log_directory, self.opts.model_type + '_' + self.opts.loss_regre)

        #   if not os.path.exists(self.log_path):
            #   os.makedirs(self.log_path)

        #   self.writer = SummaryWriter(self.log_path)
        seed = 559
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        if self.opts.multi_gpu == True:
            num_gpu = torch.cuda.device_count()
        else:
            num_gpu = 1
        
        self.path_model = os.path.join(self.opts.model_folder, self.opts.checkpoint_dir)


	    # self.network_list = {"resnet18": network.resnet18, "resnet34": network.resnet34, "resnet50": network.resnet50, "resnet101": network.resnet101}
        self.network_list = {"FF-ViT": network.FF_ViT}
        self.models = {}
        self.parameters_to_train = []
        self.results = {}
        self.test_cls = 1e6
        self.test_latent = 1e6
        self.test_cls_H = 1e6
        self.test_latent_H = 1e6
        self.epoch_cls = 0
        self.epoch_latent = 0
        self.freq_encoding = {}
        
        
        self.writer = SummaryWriter(self.path_model)

        if not self.opts.no_cuda:
            if self.opts.multi_gpu == True:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device('cuda:{}'.format(self.opts.gpu_id) if torch.cuda.is_available() else 'cpu')
        
        train_dataset = pCLE_Rotation_dataset('train', self.opts.k)
        # train_sampler = DistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.opts.batch_size, shuffle=True,
            num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)


        val_dataset = pCLE_Rotation_dataset('test', self.opts.k)
        self.val_loader = DataLoader(
            val_dataset, batch_size=41, shuffle=False,
            num_workers=self.opts.num_workers, pin_memory=True)
        
        heart_dataset = Heart_dataset()
        self.heart_loader = DataLoader(
            heart_dataset, batch_size=41, shuffle=False,
            num_workers=self.opts.num_workers, pin_memory=True
        )
        

        self.models["angle"] = self.network_list[self.opts.model_type](image_size=(self.opts.height, self.opts.width),
                                                                        patch_size=(self.opts.height_p, self.opts.width_p),
                                                                        num_classes=self.opts.out_channels,
                                                                        dim=self.opts.hidden_dim,
                                                                        depth=self.opts.depth,
                                                                        heads=self.opts.heads,
                                                                        mlp_dim=self.opts.mlp_dim,
                                                                        FFC_to_Patch=not(self.opts.no_ffpe),
                                                                        shift=self.opts.shift,
                                                                        self_locality=self.opts.self_locality,
                                                                        cross_locality=self.opts.cross_locality).to(self.device)


        num_params = count_parameters(self.models["angle"])

        print("Number of Parameters in Model: {:.1f}M".format(num_params / 1e6))
        print("Training model named:\n  ", self.opts.model_type)
        print("The {} model use {:d} hidden dimensions, {:d} layers and {:d} heads and {:d} MLP dimensions".format(self.opts.model_type,
                                                                                                                   self.opts.hidden_dim,
                                                                                                                   self.opts.depth,
                                                                                                                   self.opts.heads,
                                                                                                                   self.opts.mlp_dim))
        print("Training is using:\n  ", self.device)
        if torch.cuda.is_available():
            print('Using GPU: {} X {}'.format(num_gpu, torch.cuda.get_device_name()))

        print("Checkpoint address:", self.path_model)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        print("Using Fast Fourier Patch Embedding:", not (self.opts.no_ffpe))
        print("Using cross-attention:", not (self.opts.no_cross_attn))
        print("Using Multi-scale inference:", not(self.opts.no_msi))
        print("The batch size:", self.opts.batch_size)
        print("Number of Epochs:", self.opts.num_epochs)
        # print("Using Gaussian Prob Inference:", self.opts.use_gaussian_infer)
        # print("Using MixUp data argumentation:", self.opts.mixup)
        # print("Using pretrained:", self.opts.pretrained)
        print("Number of input channels:", self.opts.in_channels)
        print("Number of output channels:", self.opts.out_channels)
        print("{} PAR Layers".format(self.opts.num_pyrs))
        # print("Interp loss: {:.1f} * MoI + {:.1f} * SSIM + {:.1f} * BM".format(self.opts.MoI_weight, self.opts.SSIM_weight, self.opts.BM_weight))
        if self.opts.multi_gpu == True:
            self.models["angle"] = nn.DataParallel(self.models["angle"], device_ids=[0,1,2])
            
        self.model_optimizer = optim.AdamW(self.models['angle'].parameters(), self.opts.learning_rate, weight_decay=self.opts.weight_decay)

        self.L1_loss = nn.L1Loss().to(self.device)

        self.model_lr_scheduler = optim.lr_scheduler.OneCycleLR(self.model_optimizer, max_lr=1e-4, epochs=self.opts.num_epochs,
                                                                    steps_per_epoch=(len(train_dataset) // self.opts.batch_size) + 1,
                                                                    pct_start=0.1, div_factor=10, final_div_factor=0.1,three_phase=False)
        self.BM = BM().to(self.device)
        # self.BM = GDER().to(self.device)
        self.Embedding = PositionalEncoding1D(channels=self.opts.hidden_dim).to(self.device)
        self.pad = nn.ReflectionPad2d((4,4,0,0)).to(self.device)

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.opts.num_epochs):
            self.run_epoch()

    def compute_cls_accuracy(self, prob, target, N_true, N_total):
        pred = torch.argmax(prob, dim=1)
        num_true = (pred == target).sum()
        num_total = pred.shape[0]

        N_true += num_true
        N_total += num_total
        acc =  N_true / N_total

        return acc, N_true, N_total
    

    def run_epoch(self):
        #   self.model_lr_scheduler.step()
        

        print("Training")
        self.set_train()
        Train_Loader = tqdm(self.train_loader)
        Losses = {}
        Loss_latent = []
        Loss_latent_0 = []
        Loss_latent_1 = []
        Loss_latent_2 = []
        Loss_cls = []
        Loss_cls_0 = []
        Loss_cls_1 = []
        Loss_cls_2 = []
        Loss_total = []

        for batch_idx, inputs in enumerate(Train_Loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            self.batch_idx = batch_idx  
            frame = self.pad(inputs['frame'])
            if self.opts.no_cross_attn:
                BM_map = None
            else:
                BM_map = generate_BM_map(frame, H_patch=self.opts.height_p, W_patch=self.opts.width_p, metric=self.BM, embedding=self.Embedding)
            
            outputs = self.models['angle'](frame, BM_map)
            

            self.model_optimizer.zero_grad()
            Losses['l1'] = 0
            Losses['cls'] = 0
            Losses['latent'] = 0
            if self.opts.no_msi:
                angle_latent = 10 * torch.tanh(outputs["out_{}".format(2)])
                angle_cls = 10 * torch.tanh(outputs["cls_{}".format(2)])
                Losses['latent'] += self.L1_loss(angle_latent, inputs['angle'])
                Losses['cls'] += self.L1_loss(angle_cls, inputs['angle'])
                Losses['total'] = Losses['cls'] + Losses['cls']
            else:
                for i in range(self.opts.num_pyrs - 1, -1, -1):
                    # angle_latent = 10 * torch.tanh(outputs["out_{}".format(i)])
                    # angle_cls = 10 * torch.tanh(outputs["cls_{}".format(i)])

                    angle_latent = outputs["out_{}".format(i)]
                    angle_cls = outputs["cls_{}".format(i)]

                    mae_latent = self.L1_loss(angle_latent, inputs['angle'])
                    mae_cls = self.L1_loss(angle_cls, inputs['angle'])
                    Losses['latent_{}'.format(i)] = mae_latent
                    Losses['latent'] += (self.opts.pyr_decay ** i) * mae_latent       
                    Losses['cls_{}'.format(i)] = mae_cls
                    Losses['cls'] += (self.opts.pyr_decay ** i) * mae_cls       
                Losses['total'] = Losses['cls'] / self.opts.num_pyrs + Losses['latent'] / self.opts.num_pyrs
            
            Losses['total'].backward()
            self.model_optimizer.step()
            self.model_lr_scheduler.step()
            # self.model_optimizer_SeqA.step()
            self.writer.add_scalar('lr/train', np.array(self.model_lr_scheduler.get_last_lr()), self.step)
            if not(self.opts.no_msi):
                if self.opts.num_pyrs == 3:
                    Loss_latent_0.append(Losses['latent_0'].cpu().detach().numpy())
                    Loss_latent_1.append(Losses['latent_1'].cpu().detach().numpy())
                    Loss_latent_2.append(Losses['latent_2'].cpu().detach().numpy())
                    Loss_cls_0.append(Losses['cls_0'].cpu().detach().numpy())
                    Loss_cls_1.append(Losses['cls_1'].cpu().detach().numpy())
                    Loss_cls_2.append(Losses['cls_2'].cpu().detach().numpy())
            Loss_latent.append((Losses['latent'] / self.opts.num_pyrs).cpu().detach().numpy())
            Loss_cls.append((Losses['cls'] / self.opts.num_pyrs).cpu().detach().numpy())
            Loss_total.append(Losses['total'].cpu().detach().numpy())
            # Loss_ex.append(Losses['loss_Ex'].cpu().detach().numpy())

            Train_Loader.set_postfix(Loss_cls=np.mean(Loss_cls),epoch=self.epoch)

            self.step += 1

        # log training results per epoch
        if not(self.opts.no_msi):
            if self.opts.num_pyrs == 3:
                self.writer.add_scalar('Loss_train/latent_0', np.mean(Loss_latent_0), self.epoch)
                self.writer.add_scalar('Loss_train/latent_1', np.mean(Loss_latent_1), self.epoch)
                self.writer.add_scalar('Loss_train/latent_2', np.mean(Loss_latent_2), self.epoch)
                self.writer.add_scalar('Loss_train/cls_0', np.mean(Loss_cls_0), self.epoch)
                self.writer.add_scalar('Loss_train/cls_1', np.mean(Loss_cls_1), self.epoch)
                self.writer.add_scalar('Loss_train/cls_2', np.mean(Loss_cls_2), self.epoch)
        self.writer.add_scalar('Loss_train/latent', np.mean(Loss_latent), self.epoch)
        # self.writer.add_scalar('Loss_Generator/train_excess', np.mean(Loss_ex), self.epoch)
        self.writer.add_scalar('Loss_train/cls', np.mean(Loss_cls), self.epoch)
        self.writer.add_scalar('Loss_train/total', np.mean(Loss_total), self.epoch)
        # print("Testing")
        # self.val()
        print("Evaluation Metrics")
        self.test()
        
        
    def test(self):

        self.set_eval()
        Test_Loader = tqdm(self.val_loader)
        Heart_loader = tqdm(self.heart_loader)
        MAE = []
        MAE_latent = []
        MAE_cls = []
        MAE_H = []
        MAE_latent_H = []
        MAE_cls_H = []

        with torch.no_grad():
            N_true = 0
            N_total = 0
            for batch_idx, inputs in enumerate(Test_Loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)

                self.batch_idx = batch_idx  
                frame = self.pad(inputs['frame'])
                if self.opts.no_cross_attn:
                    BM_map = None
                else:
                    BM_map = generate_BM_map(frame, H_patch=self.opts.height_p, W_patch=self.opts.width_p, metric=self.BM, embedding=self.Embedding)
                outputs = self.models['angle'](frame, BM_map)

                # angle_latent = 10 * torch.tanh(outputs["out_{}".format(2)])
                # angle_cls = 10 * torch.tanh(outputs["cls_{}".format(2)])

                angle_latent = outputs["out_{}".format(self.opts.num_pyrs-1)]
                angle_cls = outputs["cls_{}".format(self.opts.num_pyrs-1)]

                mae_latent = self.L1_loss(angle_latent, inputs['angle'])
                mae_cls = self.L1_loss(angle_cls, inputs['angle'])
                mae = self.L1_loss((angle_cls + angle_latent) / 2, inputs['angle'])

                acc_dir, N_true, N_total = self.compute_dir_accuracy(angle_cls, inputs['angle'], N_true, N_total)
                
                
                MAE_latent.append(mae_latent.cpu().detach().numpy())
                MAE_cls.append(mae_cls.cpu().detach().numpy())
                MAE.append(mae.cpu().detach().numpy())
                Test_Loader.set_postfix(MAE_cls=np.mean(MAE_cls), dir_acc=acc_dir.cpu().detach().numpy(), epoch=self.epoch)
        
            self.writer.add_scalar('results_test_lens/MAE_latent', np.mean(MAE_latent), self.epoch)
            self.writer.add_scalar('results_test_lens/MAE_cls', np.mean(MAE_cls), self.epoch)
            self.writer.add_scalar('results_test_lens/MAE', np.mean(MAE), self.epoch)
            if self.test_cls > np.mean(MAE_cls):
                self.test_cls = np.mean(MAE_cls)
                self.epoch_cls = self.epoch
                self.save_checkpoint('cls')
            if self.test_latent > np.mean(MAE_latent):
                self.test_latent = np.mean(MAE_latent)
                self.epoch_latent = self.epoch
                self.save_checkpoint('latent')
            
            
            # print("The best latent MAE is {:4f} and it occurs at epoch {:d}".format(self.test_latent, self.epoch_latent))
            N_true = 0
            N_total = 0
            for batch_idx, inputs in enumerate(Heart_loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)

                self.batch_idx = batch_idx  
                frame = self.pad(inputs['frame'])
                if self.opts.no_cross_attn:
                    BM_map = None
                else:
                    BM_map = generate_BM_map(frame, H_patch=self.opts.height_p, W_patch=self.opts.width_p, metric=self.BM, embedding=self.Embedding)
                outputs = self.models['angle'](frame, BM_map)

                # angle_latent = 10 * torch.tanh(outputs["out_{}".format(2)])
                # angle_cls = 10 * torch.tanh(outputs["cls_{}".format(2)])
                angle_latent = outputs["out_{}".format(self.opts.num_pyrs-1)]
                angle_cls = outputs["cls_{}".format(self.opts.num_pyrs-1)]

                mae_latent = self.L1_loss(angle_latent, inputs['angle'])
                mae_cls = self.L1_loss(angle_cls, inputs['angle'])
                mae = self.L1_loss((angle_cls + angle_latent) / 2, inputs['angle'])

                acc_dir_H, N_true, N_total = self.compute_dir_accuracy(angle_cls, inputs['angle'], N_true, N_total)
                
                
                MAE_latent_H.append(mae_latent.cpu().detach().numpy())
                MAE_cls_H.append(mae_cls.cpu().detach().numpy())
                MAE_H.append(mae.cpu().detach().numpy())
                Heart_loader.set_postfix(MAE_cls=np.mean(MAE_cls_H), dir_acc=acc_dir_H.cpu().detach().numpy(), epoch=self.epoch)
        
            self.writer.add_scalar('results_test_heart/MAE_latent', np.mean(MAE_latent_H), self.epoch)
            self.writer.add_scalar('results_test_heart/MAE_cls', np.mean(MAE_cls_H), self.epoch)
            self.writer.add_scalar('results_test_heart/MAE', np.mean(MAE_H), self.epoch)
            if self.test_cls_H > np.mean(MAE_cls_H):
                self.test_cls_H = np.mean(MAE_cls_H)
                self.epoch_cls_H = self.epoch
            if self.test_latent_H > np.mean(MAE_latent_H):
                self.test_latent_H = np.mean(MAE_latent_H)
                self.epoch_latent_H = self.epoch
            print("The best lens cls MAE is {:4f} and it occurs at epoch {:d}".format(self.test_cls, self.epoch_cls))
            print("The best heart cls MAE is {:4f} and it occurs at epoch {:d}".format(self.test_cls_H, self.epoch_cls_H))
            print("In lens dataset and epoch {}, the MAE is {:4f}, std is {:4f}, the direction accuracy is {:4f}".format(self.epoch, np.mean(MAE_cls), np.std(MAE_cls), acc_dir.cpu().detach().numpy()))
            print("In heart dataset and epoch {}, the MAE is {:4f}, std is {:4f}, the direction accuracy is {:4f}".format(self.epoch, np.mean(MAE_cls_H), np.std(MAE_cls_H), acc_dir_H.cpu().detach().numpy()))
            # print("The best latent MAE is {:4f} and it occurs at epoch {:d}".format(self.test_latent_H, self.epoch_latent_H))




        self.set_train()


    def compute_dir_accuracy(self, pred, target, N_true, N_total):
        sign_pred = torch.sign(pred)
        sign_target = torch.sign(target)

        mask = 1 - (sign_target == 0)*1.
        num_total = mask.sum()
        num_true = ((sign_pred == sign_target) * mask).sum()

        N_true += num_true
        N_total += num_total
        acc =  N_true / N_total

        return acc, N_true, N_total
    
    def save_checkpoint(self, name):
        PATH = os.path.join(self.path_model, ('model_{}_{}.pt').format(name, self.epoch))

        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.models['angle'].state_dict(),
                    'optimizer_state_dict': self.model_optimizer.state_dict(),
                    }, PATH)


    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.path_model, ('model_{}.pt').format(epoch)))
        # input(checkpoint['model_state_dict'].keys())
        for key in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
        self.models['angle'].load_state_dict(checkpoint['model_state_dict'])
        # self.model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # input(checkpoint['model_state_dict'].keys())
        # print('train_MAE', checkpoint['train_MAE'])
        # print('val_MAE', checkpoint['val_MAE'])

    