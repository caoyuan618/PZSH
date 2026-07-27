import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


DATASETS = {
    "AWA": {
        "data_path": "dataset/AWA/JPEGImages/",
        "topK": 4000,
        "n_class": 50,
        "train_set": "dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F.txt",
        "database": "dataset/AWA/filetxt/database.txt",
        "test": "dataset/AWA/filetxt/test.txt",
    },
    "CUB": {
        "data_path": "dataset/CUB/CUB-last50_is_txt2img/images/",
        "topK": 1000,
        "n_class": 200,
        "train_set": "dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F.txt",
        "database": "dataset/CUB/CUB-last50_is_txt2img/images/database1.txt",
        "test": "dataset/CUB/CUB-last50_is_txt2img/images/test1.txt",
    },
}


def config_dataset(config):
    dataset_cfg = DATASETS[config["dataset"]]
    config["data_path"] = dataset_cfg["data_path"]
    config["topK"] = dataset_cfg["topK"]
    config["n_class"] = dataset_cfg["n_class"]
    config["data"] = {
        split: {"list_path": dataset_cfg[split], "batch_size": config["batch_size"]}
        for split in ["train_set", "database", "test"]
    }
    return config


def image_transform_for_blip():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def _resolve_path(data_path, image_path):
    return image_path if os.path.isabs(image_path) else os.path.join(data_path, image_path)


def _parse_label(label_text):
    return np.asarray([int(value) for value in label_text.split()], dtype=np.int64)


class PZSHTrainList(data.Dataset):
    def __init__(self, data_path, image_list):
        self.transform = image_transform_for_blip()
        self.samples = []
        for line in image_list:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                self.samples.append((_resolve_path(data_path, parts[0]), _parse_label(parts[1]), parts[-1]))

    def __getitem__(self, index):
        image_path, label, feature_text = self.samples[index]
        image = self.transform(Image.open(image_path).convert("RGB"))
        synthetic_feature = torch.tensor([float(value) for value in feature_text.split()], dtype=torch.float32)
        return image, label, synthetic_feature, index

    def __len__(self):
        return len(self.samples)


class PZSHImageList(data.Dataset):
    def __init__(self, data_path, image_list):
        self.transform = image_transform_for_blip()
        self.samples = []
        for line in image_list:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                self.samples.append((_resolve_path(data_path, parts[0]), _parse_label(parts[1])))

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, label, index

    def __len__(self):
        return len(self.samples)


def _read_list(list_path):
    with open(list_path, "r", encoding="utf-8") as file:
        return file.readlines()


def get_data(config):
    data_config = config["data"]
    train_set = PZSHTrainList(config["data_path"], _read_list(data_config["train_set"]["list_path"]))
    test_set = PZSHImageList(config["data_path"], _read_list(data_config["test"]["list_path"]))
    database_set = PZSHImageList(config["data_path"], _read_list(data_config["database"]["list_path"]))

    train_loader = data.DataLoader(train_set, batch_size=data_config["train_set"]["batch_size"], shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=data_config["test"]["batch_size"], shuffle=False, num_workers=4)
    database_loader = data.DataLoader(database_set, batch_size=data_config["database"]["batch_size"], shuffle=False, num_workers=4)
    return train_loader, test_loader, database_loader, len(train_set), len(test_set), len(database_set)


def compute_pzsh_result(dataloader, net, device):
    hash_codes, labels = [], []
    net.eval()
    with torch.no_grad():
        for images, batch_labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            labels.append(batch_labels)
            _, batch_hash, _ = net(images.to(device))
            hash_codes.append(batch_hash.data.cpu())
    return torch.cat(hash_codes).sign(), torch.cat(labels)


def CalcHammingDist(B1, B2):
    bit = B2.shape[1]
    return 0.5 * (bit - np.dot(B1, B2.transpose()))


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0.0
    for i in tqdm(range(num_query), desc="Calculating mAP"):
        relevant = (np.dot(queryL[i, :], retrievalL.transpose()) > 0).astype(np.float32)
        relevant = relevant[np.argsort(CalcHammingDist(qB[i, :], rB))][:topk]
        relevant_count = int(np.sum(relevant))
        if relevant_count == 0:
            continue
        precision_index = np.asarray(np.where(relevant == 1)) + 1.0
        precision_count = np.linspace(1, relevant_count, relevant_count)
        topkmap += np.mean(precision_count / precision_index)
    return topkmap / num_query
