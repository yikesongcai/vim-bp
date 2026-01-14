import os
import shutil
import urllib.request
import zipfile
from typing import Any
import math
import json
import numpy as np
from torchvision.datasets import MNIST
from PIL import Image
import torch
import torchvision
import torchvision.datasets.utils as tdu
from tqdm import tqdm

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.9637,), (0.1591,))])
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'MNIST')

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url:urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, root:str, train:bool=True, transform:Any=None, target_transform:Any=None,
                 download:bool=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        if not os.path.exists(os.path.join(self.processed_folder, data_file)):
            self.download_and_process()
        self.data, self.targets, self.user_index = torch.load(os.path.join(self.processed_folder, data_file))
        self.id = self.user_index.tolist()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download_and_process(self):
        """
        Download the raw data and process it and save it in Torch format
        Modified from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/cv/dataset/leaf_cv.py
        """
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        if not os.path.exists(os.path.join(self.raw_folder, 'all_data')):
            """Download the FEMNIST data if it doesn't exist in processed_folder already."""
            # Download to `self.raw_dir`.
            url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
            name = 'femnist_all_data.zip'
            tdu.download_and_extract_archive(f'{url}/{name}', self.raw_folder, remove_finished=True)
            # src_path = download_from_url(f'{url}/{name}', os.path.join(, 'femnist_all_data.zip'))
            # tar_paths = extract_from_zip(src_path, self.raw_folder)

            # with open(os.path.join(self.raw_folder, 'all_data.json'), 'w') as f:
            #     json.dump(all_data, f)
            # os.remove(src_path)
            # shutil.rmtree(tar_paths[0])
        # else:
        #     if not os.path.exists(os.path.join(self.raw_folder, 'all_data')):
        #         tar_paths = extract_from_zip(os.path.join(self.raw_folder, 'femnist_all_data.zip'), self.raw_folder)
        #     else:
        tar_path0 = os.path.join(self.raw_folder, 'all_data')
        tar_paths = [tar_path0] + [os.path.join(tar_path0, r) for r in os.listdir(tar_path0)]
            # with open(os.path.join(self.raw_folder, 'all_data.json'), 'r') as f:
            #     all_data = json.load(f)
        all_data = {
            'users': [],
            'num_samples': [],
            'user_data': {}
        }
        for i in tqdm(range(1, len(tar_paths)), desc='Loading data'):
            with open(tar_paths[i], 'r') as f:
                raw_data = json.load(f)
                all_data['users'].extend(raw_data['users'])
                all_data['num_samples'].extend(raw_data['num_samples'])
                all_data['user_data'].update(raw_data['user_data'])
        """Process Data"""
        Xs = []
        Ys = []
        sample_ids = []
        for idx, (writer, v) in enumerate(all_data['user_data'].items()):
            data, targets = v['x'], v['y']
            Xs.extend(data)
            Ys.extend(targets)
            sample_ids.extend([idx] * len(data))
        Xs = torch.tensor(np.stack(Xs))/255.0
        Ys = torch.LongTensor(np.stack(Ys))
        sample_ids = torch.tensor(np.stack(sample_ids))
        num_samples = sample_ids.shape[0]
        s1 = int(num_samples * 0.9)
        s2 = num_samples - s1
        torch.manual_seed(0)
        train_ids, test_ids = torch.utils.data.random_split(sample_ids, [s1, s2])
        train_indices = train_ids.indices
        test_indices = test_ids.indices
        train_data, train_targets, train_sample_id = Xs[train_indices], Ys[train_indices], sample_ids[train_indices]
        test_data, test_targets, test_sample_id = Xs[test_indices], Ys[test_indices], sample_ids[test_indices]
        torch.save((train_data, train_targets, train_sample_id), os.path.join(self.processed_folder, "training.pt"))
        torch.save((test_data, test_targets, test_sample_id), os.path.join(self.processed_folder, "test.pt"))

train_data = FEMNIST(root=root, download=True, train=True, transform=transform)
test_data = FEMNIST(root=root, download=True, train=False, transform=transform)