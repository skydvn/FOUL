## to generate the PACS dataset

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import tarfile
from zipfile import ZipFile
import gdown
from PIL import Image
from os import path
import json

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "PACS/"
data_path = "PACS/"


class PACS(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(PACS, self).__init__()
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
    train_dataset = PACS(train_data_paths, train_data_labels, transforms_train, domain_name)
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataset = PACS(test_data_paths, test_data_labels, transforms_test, domain_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_loader, test_loader


def separate_domain_data_simple(dataset, num_clients, num_classes, num_domains,
                                 niid=False, balance=False, partition=None):
    """
    Simple version that works with existing PFLlib utils.
    Distributes clients evenly across domains.
    """
    dataset_image, dataset_label = dataset
    
    X = {i: [] for i in range(num_clients)}
    y = {i: [] for i in range(num_clients)}
    statistic = {i: {} for i in range(num_clients)}
    domain_map = {}
    
    # Distribute clients evenly across domains
    clients_per_domain = num_clients // num_domains
    
    print(f"\nDistributing {num_clients} clients across {num_domains} domains")
    print(f"Clients per domain: {clients_per_domain}")
    
    for domain_id in range(num_domains):
        start_client = domain_id * clients_per_domain
        end_client = start_client + clients_per_domain
        if domain_id == num_domains - 1:
            end_client = num_clients
        
        domain_clients = list(range(start_client, end_client))
        
        for client_id in domain_clients:
            domain_map[client_id] = domain_id
        
        print(f"Domain {domain_id}: clients {start_client} to {end_client-1}")
        
        domain_data = dataset_image[domain_id]
        domain_labels = dataset_label[domain_id]
        
        # Distribute data
        if balance:
            samples_per_client = len(domain_data) // len(domain_clients)
            for i, client_id in enumerate(domain_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                if i == len(domain_clients) - 1:
                    end_idx = len(domain_data)
                
                X[client_id] = domain_data[start_idx:end_idx]
                y[client_id] = domain_labels[start_idx:end_idx]
        else:
            # Random split
            indices = np.random.permutation(len(domain_data))
            samples_per_client = len(domain_data) // len(domain_clients)
            
            for i, client_id in enumerate(domain_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                if i == len(domain_clients) - 1:
                    end_idx = len(domain_data)
                
                client_indices = indices[start_idx:end_idx]
                X[client_id] = domain_data[client_indices]
                y[client_id] = domain_labels[client_indices]
    
    # Compute statistics
    for client_id in range(num_clients):
        unique_labels = np.unique(y[client_id])
        statistic[client_id] = {
            'num_samples': len(y[client_id]),
            'domain': domain_map[client_id],
            'labels': unique_labels.tolist()
        }
    
    return X, y, statistic, domain_map


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    root = data_path + "rawdata/Homework3-PACS"
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']

    # Get PACS data
    if not os.path.exists(root):
        os.makedirs(data_path + "rawdata")
        os.system(f'cd {data_path}rawdata && git clone https://github.com/MachineLearning2020/Homework3-PACS.git')

    # Load data for each domain
    dataset_image = []
    dataset_label = []
    class_unique = []
    
    for id, d in enumerate(domains):
        train_loader, test_loader = get_pacs_dloader(root, d)
        for _, tt in enumerate(train_loader):
            train_data, train_label = tt
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt
        
        dataset_image_d = []
        dataset_label_d = []
        dataset_image_d.extend(train_data.cpu().detach().numpy())
        dataset_image_d.extend(test_data.cpu().detach().numpy())
        dataset_label_d.extend(train_label.cpu().detach().numpy())
        dataset_label_d.extend(test_label.cpu().detach().numpy())

        dataset_image_d = np.array(dataset_image_d)
        dataset_label_d = np.array(dataset_label_d)

        dataset_image.append(dataset_image_d)
        dataset_label.append(dataset_label_d)
        
        print(f"Domain {id} ({d}): {len(dataset_label_d)} samples")
        class_unique.extend(set(dataset_label[id]))

    num_classes = len(set(class_unique))
    print(f'Number of classes: {num_classes}')
    
    # Separate data by domain
    X, y, statistic, domain_map = separate_domain_data_simple(
        (dataset_image, dataset_label), 
        num_clients, 
        num_classes, 
        len(domains),
        niid, 
        balance, 
        partition
    )

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, 
              num_clients, num_classes, statistic, niid, balance, partition)
    
    # Save domain mapping
    domain_config_path = dir_path + "domain_config.json"
    with open(domain_config_path, 'w') as f:
        json.dump({
            'domain_names': domains,
            'domain_map': domain_map,
            'clients_per_domain': {
                domains[i]: [cid for cid, did in domain_map.items() if did == i]
                for i in range(len(domains))
            }
        }, f, indent=4)
    
    print(f"\nDomain configuration saved to {domain_config_path}")
    print("Dataset generation complete!")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
