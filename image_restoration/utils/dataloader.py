import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PairedImageDataset(Dataset):
    def __init__(self, dataset_path, dataroot_lq, dataroot_gt, subset='train'):
        self.dataroot_lq = os.path.join(dataset_path, dataroot_lq)
        self.dataroot_gt = os.path.join(dataset_path, dataroot_gt)
        self.subset = subset
        config_file = os.path.join(dataset_path, 'dataset_metadata.yaml')
        
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.dataset_len = self.config[f'{subset}_set_size']
        self.load_dataset()
        
    def load_dataset(self):
        self.lq = []
        self.gt = []
        print(f'Loading {self.subset} dataset...')
        for i in tqdm(range(self.dataset_len)):
            self.lq.append(torch.from_numpy(np.load(os.path.join(self.dataroot_lq, f'{i}.npy'))).float().unsqueeze(0))
            self.gt.append(torch.from_numpy(np.load(os.path.join(self.dataroot_gt, f'{i}.npy'))).float().unsqueeze(0))
            
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return self.lq[idx], self.gt[idx]
    
def get_dataloader_PID(dataset_path: str, dataroot_lq: str, dataroot_gt: str, batch_size: int, num_workers: int, subset='train') -> DataLoader:
    """ get dataloader for PairedImageDataset

    Args:
        dataset_path (str): dataset path
        dataroot_lq (str): dataroot of lq (low quality images)
        dataroot_gt (str): dataroot of gt (ground truth images)
        batch_size (int): batch size
        num_workers (int): number of workers when loading data
        subset (str, optional): 'train', 'val' or 'test'. Defaults to 'train'.

    Returns:
        DataLoader: dataloader
    """
    
    dataset = PairedImageDataset(dataset_path, dataroot_lq, dataroot_gt, subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


def load_dataset(config: dict, batch_size: int, num_workers: int) -> tuple:
    """ load dataset

    Args:
        config (dict): configuration
        numworkers (int): number of workers when loading data

    Raises:
        NotImplementedError: dataset type is not implemented

    Returns:
        tuple: train_loader, val_loader
    """
    dataset_type = config['datasets']['type']
    if dataset_type == 'PairedImageDataset':
        train_dataset_root = config['datasets']['train']['dataset_root']
        train_dataset_lq = config['datasets']['train']['dataroot_lq']
        train_dataset_gt = config['datasets']['train']['dataroot_gt']
        
        val_dataset_root = config['datasets']['val']['dataset_root']
        val_dataset_lq = config['datasets']['val']['dataroot_lq']
        val_dataset_gt = config['datasets']['val']['dataroot_gt']
        
        train_loader = get_dataloader_PID(train_dataset_root, train_dataset_lq, train_dataset_gt, batch_size, num_workers, 'train')
        val_loader = get_dataloader_PID(val_dataset_root, val_dataset_lq, val_dataset_gt, batch_size, num_workers, 'val')
        
        return train_loader, val_loader
        
    else:
        raise NotImplementedError(f'Unsupported dataset type: {dataset_type}')


if __name__ == '__main__':
    train_loader = get_dataloader_PID('/data1/zyzhao/datasets/GainMat-F16-sob-inf100', 'train/lq', 'train/gt', 64, 4, 'train')
    val_loader = get_dataloader_PID('/data1/zyzhao/datasets/GainMat-F16-sob-inf100', 'val/lq', 'val/gt', 64, 4, 'val')
    
    for data in train_loader:
        lq, gt = data
        print(lq.shape, gt.shape)
        break
        
