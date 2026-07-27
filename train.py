import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from A_utils.tools_with_category_name import CalcTopMap, compute_pzsh_result, config_dataset, get_data
from loss import PZSHAlignmentLoss, PZSHProxyHashLoss
from model import PZSH


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="PZSH training")
    parser.add_argument("--dataset", type=str, choices=["AWA", "CUB"], default="AWA")
    parser.add_argument("--bits", type=int, nargs="+", default=[24, 48, 64, 128])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="outputs/PZSH")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_config(args):
    blip_main_dir = os.environ.get("BLIP_MAIN_DIR", os.path.join(PROJECT_DIR, "BLIP_main"))
    config = {
        "dataset": args.dataset,
        "net": PZSH,
        "device": torch.device(args.device),
        "bit_list": args.bits,
        "batch_size": args.batch_size,
        "epoch": args.epochs,
        "test_map": 2,
        "save_path": args.save_path,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        "lambda_quant": 1e-4,
        "proxy_warmup": 10,
        "momentum": 0.9,
        "blip_pretrained_pth": os.path.join(blip_main_dir, "models", "BLIP_base.pth"),
        "blip_med_config": os.path.join(blip_main_dir, "configs", "med_config.json"),
        "blip_vit_mode": "base",
    }

    if args.dataset == "AWA":
        config.update({"n_class": 50, "num_train": 4400, "alpha": 0.5, "beta": 1.0})
    else:
        config.update({"n_class": 200, "num_train": 6000, "alpha": 0.1, "beta": 0.7})

    return config_dataset(config)


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, _, _ = get_data(config)
    config["num_train"] = num_train

    net = config["net"](config, bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **config["optimizer"]["optim_params"])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    alignment_loss_fn = PZSHAlignmentLoss(temperature=0.07)
    hash_loss_fn = PZSHProxyHashLoss(config, bit)
    best_mAP = 0.0

    for epoch in range(config["epoch"]):
        net.train()
        train_loss = 0.0
        alignment_loss_sum = 0.0
        hash_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epoch']}", ncols=120)
        for images, labels, synthetic_features, indices in pbar:
            images = images.to(device)
            labels = labels.to(device).float()
            synthetic_features = synthetic_features.to(device)

            optimizer.zero_grad()
            real_features, query_hash, momentum_hash = net(images)
            alignment_loss = alignment_loss_fn(real_features, synthetic_features, labels)
            hash_loss = hash_loss_fn(query_hash, momentum_hash, labels, indices, epoch)
            loss = alignment_loss + config["alpha"] * hash_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            alignment_loss_sum += alignment_loss.item()
            hash_loss_sum += config["alpha"] * hash_loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "align": f"{alignment_loss.item():.4f}",
                "hash": f"{config['alpha'] * hash_loss.item():.4f}",
            })

        scheduler.step()
        train_loss /= len(train_loader)
        alignment_loss_sum /= len(train_loader)
        hash_loss_sum /= len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}: loss={train_loss:.6f}, "
            f"alignment={alignment_loss_sum:.6f}, hash={hash_loss_sum:.6f}"
        )

        if (epoch + 1) % config["test_map"] == 0:
            tst_binary, tst_label = compute_pzsh_result(test_loader, net, device)
            trn_binary, trn_label = compute_pzsh_result(dataset_loader, net, device)
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
            logger.info(f"Epoch {epoch + 1}: mAP={mAP:.4f}, best={best_mAP:.4f}")

            if mAP > best_mAP:
                best_mAP = mAP
                np.save(os.path.join(config["save_path"], f"{config['dataset']}_best_tst_binary_bit{bit}.npy"), tst_binary.numpy())
                np.save(os.path.join(config["save_path"], f"{config['dataset']}_best_tst_label_bit{bit}.npy"), tst_label.numpy())
                np.save(os.path.join(config["save_path"], f"{config['dataset']}_best_trn_binary_bit{bit}.npy"), trn_binary.numpy())
                np.save(os.path.join(config["save_path"], f"{config['dataset']}_best_trn_label_bit{bit}.npy"), trn_label.numpy())


def main():
    args = parse_args()
    setup_seed(args.seed)
    config = get_config(args)
    os.makedirs(config["save_path"], exist_ok=True)
    logger.add(os.path.join(config["save_path"], f"{config['dataset']}_PZSH_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    logger.info("Configuration:\n" + json.dumps(config, indent=4, default=str))

    for bit in config["bit_list"]:
        logger.info(f"Start training: dataset={config['dataset']}, bit={bit}")
        train_val(config, bit)


if __name__ == "__main__":
    main()
