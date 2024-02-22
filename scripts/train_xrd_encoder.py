
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

from torch_geometric.data import DataLoader

from cdvae.pl_modules.xrd_encoder import XRDEncoder
from cdvae.pl_data.dataset import CrystXRDDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from scripts.eval_utils import load_model

class XRDTrainer:
    def __init__(self, data_dir, save_file, model_path, batch_size, epochs, lr):
        
        self.data_dir = data_dir
        self.save_file = save_file
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_data()
        self.load_teacher_model()
        self.create_enc_model()
        self.optimizer = optim.Adam(self.enc_model.parameters(), lr=self.lr)

    def load_data(self):
        # train loader
        data_path = self.data_dir
        train_dataset = CrystXRDDataset(
            data_path,
            filename='train.csv',
        )
        scaler = get_scaler_from_data_list(
            train_dataset.cached_data,
            key=train_dataset.prop
        )
        train_dataset.scaler = scaler
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )
        # val loader
        val_dataset = CrystXRDDataset(
            data_path,
            filename='val.csv',
        )
        scaler = get_scaler_from_data_list(
            val_dataset.cached_data,
            key=val_dataset.prop
        )
        val_dataset.scaler = scaler
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def load_teacher_model(self):
        self.teacher_model, _, _ = load_model(Path(self.model_path))

    def create_enc_model(self):
        self.enc_model = XRDEncoder()
        self.enc_model.to(self.device)

    def train(self):
        val_loss_min = float('inf')
        for epoch in range(self.epochs):
            self.train_epoch()
            val_loss = self.eval()
            if val_loss < val_loss_min:
                print(f'Epoch {epoch}: Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.save(self.enc_model.state_dict)
                val_loss_min = val_loss
            else:
                print(f'Epoch {epoch}: Validation loss: {val_loss:.6f}')

    def train_epoch(self):
        self.enc_model.train()
        running_loss = 0.0
        for (data, xrd) in self.train_loader:
            data = data.to(self.device)
            xrd = xrd.to(self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            pred_embedding = self.enc_model(xrd)
            with torch.no_grad():
                teacher_embedding_mu, _, _ = self.teacher_model.encode(data)
            loss = F.mse_loss(pred_embedding, teacher_embedding_mu)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return running_loss / len(self.train_loader)

    def eval(self):
        self.enc_model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for (data, xrd) in self.val_loader:
                data = data.to(self.device)
                xrd = xrd.to(self.device).unsqueeze(1)
                pred_embedding = self.enc_model(xrd)
                teacher_embedding_mu, _, _ = self.teacher_model.encode(data)
                loss = F.mse_loss(pred_embedding, teacher_embedding_mu)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)
    
    def save(self, state_dict):
        torch.save(state_dict, os.path.join(self.model_path, f'{self.save_file}.pt'))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train the XRD encoder to predict latent molecule representations')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='number of epochs to train (default: 10)'
    )                   
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=256, 
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--model_path',
        default='/home/tsaidi/Research/cdvae_xrd/hydra/singlerun/2024-02-17/perov',
        type=str,
    )
    parser.add_argument(
        '--data_dir',
        default='/home/tsaidi/Research/cdvae_xrd/data/perov_5',
        type=str,
    )
    parser.add_argument(
        '--save_file',
        default='xrd_enc',
        type=str,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='learning rate'
    )
    args = parser.parse_args()
    trainer = XRDTrainer(**vars(args))
    trainer.train()

if __name__ == "__main__":
    main()