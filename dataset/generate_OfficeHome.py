import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, separate_domain_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import tarfile
from zipfile import ZipFile
import gdown
from PIL import Image
from os import path
import requests


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "OfficeHome/"
data_path = "OfficeHome/"


class OfficeHome(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(OfficeHome, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)
    
    
    
def read_pacs_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels



def get_pacs_dloader(dataset_path, domain_name):
    train_data_paths, train_data_labels = read_pacs_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_pacs_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    train_dataset = OfficeHome(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = OfficeHome(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


    
def download_OfficeHome_dataset():
    url = 'https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg'
    output = 'OfficeHome.tar.gz'

    ## download the zip file
    gdown.download(url, output, quiet=False)
    
    
    
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    ## setting up the directory for test and train and json files to be split
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata/"
    
    domains = ['Art', 'Clipart', 'Product', 'Real-World']
    
    ## getting the vlcs dataset
    if not os.path.exists(root):
        os.makedirs(root)
        download_OfficeHome_dataset()
        os.system(f'unzip VLCS.tar.gz -d {root}')
        
    
if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)