from A_utils.tools_with_category_name import *

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
blip_main_models_path = os.path.join(current_path, 'BLIP_main')
sys.path.insert(0, blip_main_models_path)

print(f"sys.path = {sys.path}")

import torch
import torch.optim as optim
import time
import numpy as np
from loguru import logger

import torch.nn.functional as F
import random
from scipy.linalg import hadamard
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
from math import gamma
# from numpy.lib.function_base import select
import torch.nn as NN
from scipy.linalg import hadamard, eig

from tqdm import tqdm  # 可视化

import torch
import torch.nn as nn
import torch.nn.functional as F
from BLIP_main.models import blip_itm
from torch.optim.lr_scheduler import StepLR

from scipy.special import comb
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="BLIP Hash Model Training")

    parser.add_argument('--dataset', type=str, choices=['AWA', 'CUB'], default="AWA", help="Dataset to use")
    parser.add_argument('--caption', type=int, default=1)
    parser.add_argument('--TGI', type=int, default=1)

    parser.add_argument('--freeze_img', type=int, default=1)
    parser.add_argument('--freeze_txt', type=int, default=1)

    parser.add_argument('--blip_loss', type=str, default= "-logsoftmax")
    parser.add_argument('--hash_loss', type=str, default="center_loss")
    parser.add_argument('--inform', type=str, default = "PZSH")

    parser.add_argument('--save_path', type=str)

    return parser.parse_args()


def get_config(args):
    config = {
        'inform': args.inform,
        'freeze_img_encoder': args.freeze_img,
        'freeze_txt_encoder': args.freeze_txt,

        "TGI": args.TGI,
        "blip_loss": args.blip_loss,
        "hash_loss": args.hash_loss,

        "net": Blip_Hash_NET,

        "dataset": args.dataset,
        "caption": bool(args.caption),

        'without_BN': 1, 

        'lambda': 0.0001,
        'lambda1': 1, 
        'lambda2': 0.0001,     
        'beta': 1,            
        'epoch_change': 10,   
        'mome': 0.9,

        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[Blip_Hash_768_3]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,  

        "epoch": 25,
        "test_map": 2,
        "save_path": args.save_path,
        "device": torch.device("cuda:0"),
        "bit_list": [64],

        "blip_pretrained_pth": 'BLIP_main/models/BLIP_base.pth',
        "blip_med_config": 'BLIP_main/configs/med_config.json',
        "blip_vit_mode": 'base',
    }
    if config["dataset"] == "AWA": 
        config["txt2img_n"] = 100
        config["n_class"] = 50
        config["num_train"] = 5000
    elif config["dataset"] == "CUB": 
        config["txt2img_n"] = 40
        config["n_class"] = 200
        config["num_train"] = 6500  
    config = config_dataset(config)
    return config

class Blip_Hash(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.hash_bit = bit

        self.blip = blip_itm.blip_itm(
            pretrained=config["blip_pretrained_pth"],   
            med_config=config["blip_med_config"],         
            image_size=224,
            vit=config["blip_vit_mode"],             
        )
        self.visual_encoder = self.blip.visual_encoder
        self.vision_proj = self.blip.vision_proj

        self.hash_head = nn.Sequential(
            nn.Linear(768, 512),     
            nn.ReLU(),
            nn.Linear(512, self.hash_bit)
        )

    def forward(self, x):
        vision_embeds = self.visual_encoder(x)
        cls_feat = vision_embeds[:, 0, :]

        proj_feat = F.normalize(cls_feat, dim=-1)
        

        hash_feat = self.hash_head(cls_feat)
        hash_feat = torch.tanh(hash_feat) 
        return proj_feat, hash_feat

class Blip_Hash_NET(nn.Module):
    def __init__(self, config, hash_bit):
        super(Blip_Hash_NET, self).__init__()
        self.m = config['mome']
        self.encoder_q = Blip_Hash(config, hash_bit)
        self.encoder_k = Blip_Hash(config, hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False 
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        blip_f1, encode_x= self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            blip_f2, encode_x2 = self.encoder_k(x)
        return blip_f1, encode_x, encode_x2

class LogSoftmaxContrastiveLoss_all_positive_samples(nn.Module):
    def __init__(self, temperature=0.07, exclude_self=True):

        super().__init__()
        self.temperature = temperature
        self.exclude_self = exclude_self
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred_feats, target_feats, labels):
        
        pred_feats = F.normalize(pred_feats, dim=-1)
        target_feats = F.normalize(target_feats, dim=-1)

        
        logits = torch.matmul(pred_feats, target_feats.T) / self.temperature
        log_probs = self.log_softmax(logits)  # [B, B]

     
        with torch.no_grad():

            pos_mask = (labels @ labels.T)  
            pos_mask = (pos_mask > 0).float()

            if self.exclude_self:
                pos_mask.fill_diagonal_(0.0)


            pos_count = pos_mask.sum(dim=1, keepdim=True)
            pos_mask = pos_mask / (pos_count + 1e-6)

        loss = - (pos_mask * log_probs).sum(dim=1).mean()

        return loss


def train_val(config, bit):
    
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    net = config["net"](config, bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **config["optimizer"]["optim_params"])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)  # 每10个epoch将lr减半

    cosine_loss_fn = LogSoftmaxContrastiveLoss_all_positive_samples(temperature=0.07)

    l = list(range(config['n_class']))

    hash_criterion2 = Center_Loss1(config, bit, l)


    best_mAP = 0

    for epoch in range(config["epoch"]):
        net.train()
        train_loss = 0
        align_loss_sum = 0
        hash_loss_sum1 = 0
        hash_loss_sum2 = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epoch']} Training", ncols=150)

        for images, label_onehot, BLIP_target, ind in pbar:
            images = images.to(device)
            label_onehot = label_onehot.to(device).float()
            BLIP_target = BLIP_target.to(device)

            optimizer.zero_grad()

            pred_features, pred_hash, pred_hash2 = net(images)

            align_loss = cosine_loss_fn(pred_features, BLIP_target, label_onehot, )


            h_loss2 = hash_criterion2(pred_hash, pred_hash2, label_onehot, ind, epoch) 
            weight2 = 1 
           
            loss = align_loss +  weight2 * h_loss2
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            align_loss_sum += align_loss.item()

            hash_loss_sum2 += weight2 * h_loss2.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'align_loss': f'{align_loss.item():.4f}',
              
                'hash_loss2': f'{weight2 * h_loss2.item():.4f}',
            })

        scheduler.step()
        train_loss = train_loss / len(train_loader)
        align_loss_sum = align_loss_sum / len(train_loader)
        hash_loss_sum1 = hash_loss_sum1 / len(train_loader)
        hash_loss_sum2 = hash_loss_sum2 / len(train_loader)

        print(f"[Epoch {epoch+1}] Loss: {train_loss:.6f} | Align Loss: {align_loss_sum:.6f} | Hash Loss1: {hash_loss_sum1:.6f} | Hash Loss2: {hash_loss_sum2:.6f}")
        logger.info(f"[Epoch {epoch+1}] Loss: {train_loss:.6f} | Align Loss: {align_loss_sum:.6f} | Hash Loss1: {hash_loss_sum1:.6f} | Hash Loss2: {hash_loss_sum2:.6f}")

        # 测试mAP
        if (epoch + 1) % config["test_map"] == 0:
            with torch.no_grad():
                tst_binary, tst_label = compute_result_BlipHash1(test_loader, net, device)
                trn_binary, trn_label = compute_result_BlipHash1(dataset_loader, net, device)
    
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
            print(f"[Eval] Epoch {epoch+1} mAP: {mAP:.4f} Best mAP: {best_mAP:.4f}")
            logger.info(f"[Eval] Epoch {epoch+1} mAP: {mAP:.4f} Best mAP: {best_mAP:.4f}")

            if mAP > best_mAP:
                best_mAP = mAP

                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_tst_binary_bit{bit}.npy"),
                    tst_binary.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_tst_label_bit{bit}.npy"),
                    tst_label.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_trn_binary_bit{bit}.npy"),
                    trn_binary.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_trn_label_bit{bit}.npy"),
                    trn_label.numpy()
                )

                logger.info(f"✅ Best model saved with mAP: {mAP:.4f}")



if __name__ == "__main__":
    args = parse_args()
    config = get_config(args)

    dataset_name = f"{config['dataset']}"
    logger.add(f"{config['save_path']}/" + dataset_name  + config['blip_loss'] +"--" + config['hash_loss'] + "--" + config["inform"] + f'{time.ctime()}.log')
    import json
    logger.info("config：\n" + json.dumps(config, indent=4, ensure_ascii=False, default=str))


    for bit in config["bit_list"]:

        logger.info(f"{config['inform']}")
        logger.info(f"{config['dataset']}") 
        logger.info(f"{bit}")
        logger.info(f"blip_loss：{config['blip_loss']}，  hash_loss：{config['hash_loss']}")
        
        train_val(config, bit)

    #  pip install transformers==4.36.2
