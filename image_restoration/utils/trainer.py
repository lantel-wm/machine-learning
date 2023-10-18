import os
import yaml
import json
import torch
import mlflow
import argparse

from torch import nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchkeras import summary
from tqdm import tqdm
from utils.dataloader import load_dataset
from models import model_factory

class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.num_workers = args.num_workers
        self.data = args.data
        self.device = args.device
        self.path = args.path
        self.name = args.name
        self.resume = args.resume
        self.early_stop = args.early_stop
        with open(self.data, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
        self.__verbose_settings()
        self.__set_trainer()
        self.__make_exp_dirs()
    
    
    # public method
    def train(self) -> None:
        """ train model
        """
        with mlflow.start_run():
            # mlflow.create_experiment(self.name)
            mlflow.autolog()
            mlflow.log_param('lr', self.optimizer.param_groups[0]['lr'])
            min_loss = 1e10
            min_loss_epoch = 0
            patience = self.early_stop
            for epoch in range(self.epochs):
                loop = tqdm(enumerate(self.train_loader), desc=f'Trainning...   Epoch {epoch}/{self.epochs}', total=len(self.train_loader))
                self.model.train()
                for i, data in loop:
                    lq, gt = data
                    lq, gt = lq.to(self.device), gt.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(lq)
                    loss = self.criterion(pred, gt)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    loop.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
                    
                mlflow.log_metric('train_MSE', loss, step=epoch)
                mlflow.log_metric('lr', self.optimizer.param_groups[0]['lr'], step=epoch)
                
                loop = tqdm(enumerate(self.val_loader), desc=f'        Validating...', total=len(self.val_loader))
                self.model.eval()
                is_best = False
                for i, data in loop:
                    lq, gt = data
                    lq, gt = lq.to(self.device), gt.to(self.device)
                    pred = self.model(lq)
                    loss = self.criterion(pred, gt)
                    loop.set_postfix(loss=loss.item())
                    if loss.item() < min_loss:
                        is_best = True
                        min_loss = loss.item()
                        min_loss_epoch = epoch
                        
                if patience != 0 and epoch - min_loss_epoch > patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                        
                mlflow.log_metric('val_MSE', loss, step=epoch)
                self.__save_model(epoch, is_best)
    
    
    # private method
    def __verbose_settings(self) -> None:
        """ print settings
        """
        print('\n\nTrain settings:\n\n')
        for arg in vars(self.args):
            print(f'{arg}: {getattr(self.args, arg)}')
        for key in self.config:
            if key == 'datasets':
                print(f'{key}:')
                print(json.dumps(self.config[key], indent=4))
            else:
                print(f'{key}: {self.config[key]}')
        print('\n\n')
        
        
    def __set_trainer(self) -> None:
        """ set trainer
        """
        self.parallel = False
        if self.device == 'cpu':
            self.device = torch.device('cpu')
        elif self.device == 'parallel':
            self.parallel = True
            self.device = torch.device('cuda')
        elif self.device in ['0', '1', '2', '3']:
            self.device = torch.device(f'cuda:{self.device}')
            
        # set dataloader
        self.train_loader, self.val_loader = load_dataset(self.config, self.batch_size, self.num_workers)
        
        # set model
        self.model = model_factory(self.config['model'])
        summary(self.model, input_shape=(1, 960, 240))
        self.model = self.model.to(self.device)
        if self.parallel == True:
            self.model = nn.DataParallel(self.model)
        if self.resume != '':
            self.model.load_state_dict(torch.load(self.resume))
            
        # set optimizer, scheduler, criterion
        self.optimizer = getattr(torch.optim, self.config['optimizer']['name'])(self.model.parameters(), **self.config['optimizer']['args'])
        self.scheduler = getattr(torch.optim.lr_scheduler, self.config['scheduler']['name'])(self.optimizer, **self.config['scheduler']['args'])
        self.criterion = getattr(nn, self.config['criterion']['name'])(**self.config['criterion']['args'])
        
        
    def __increment_exp_dir(self, exp_dir_root: str) -> str:
        """ increment experiment directory

        Args:
            exp_dir_root (str): experiment directory root
        """
        i = 2
        while True:
            exp_dir = f'{exp_dir_root}_{i}'
            if not os.path.exists(exp_dir):
                return exp_dir
            i += 1
        
        
    def __make_exp_dirs(self):
        """ make experiment directories
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        exp_dir_root = os.path.join(self.path, self.name)
        if not os.path.exists(exp_dir_root):
            os.makedirs(exp_dir_root)
        else:
            exp_dir_root = self.__increment_exp_dir(exp_dir_root)
            os.makedirs(exp_dir_root)
        
        self.exp_dir_root = exp_dir_root
        self.model_dir = os.path.join(self.exp_dir_root, 'weights')
        os.makedirs(self.model_dir)
        print(f'\n\nExperiment directory: {self.exp_dir_root}\n\n')
        
        
    def __save_model(self, epoch:int, is_best:bool) -> None:
        """ save model
        Args:
            epoch (int): epoch
            is_best (bool): is best model
        """
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'{epoch + 1}.pt'))
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'best.pt'))