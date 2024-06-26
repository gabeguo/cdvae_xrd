
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

from torch_geometric.data import DataLoader

from cdvae.pl_modules.xrd import XRDEncoder, XRDRegressor
from cdvae.pl_data.dataset import CrystXRDDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from scripts.eval_utils import load_model

import wandb

dataset_to_prop = {
    'perov_5': 'heat_ref',
    'mp_20': 'formation_energy_per_atom',
    'carbon_24': 'energy_per_atom',
}

class XRDEncoderTrainer:
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
        prop = dataset_to_prop[self.data_dir.split('/')[-1]]
        # train loader
        data_path = self.data_dir
        train_dataset = CrystXRDDataset(
            data_path,
            filename='train.csv',
            prop=prop,
        )
        train_dataset.lattice_scaler = torch.load(
            Path(self.model_path) / 'lattice_scaler.pt')
        train_dataset.scaler = torch.load(Path(self.model_path) / 'prop_scaler.pt')
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
            prop=prop,
        )
        val_dataset.lattice_scaler = torch.load(
            Path(self.model_path) / 'lattice_scaler.pt')
        val_dataset.scaler = torch.load(Path(self.model_path) / 'prop_scaler.pt')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )
        # test loader
        test_dataset = CrystXRDDataset(
            data_path,
            filename='test.csv',
            prop=prop,
        )
        test_dataset.lattice_scaler = torch.load(
            Path(self.model_path) / 'lattice_scaler.pt')
        test_dataset.scaler = torch.load(Path(self.model_path) / 'prop_scaler.pt')
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def load_teacher_model(self):
        self.teacher_model, _, _ = load_model(Path(self.model_path))
        self.teacher_model.eval()
        
    def create_enc_model(self):
        self.enc_model = XRDEncoder()
        self.enc_model.to(self.device)

    def train(self):
        val_loss_min = float('inf')
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, _ = self.eval(the_loader=self.val_loader)
            if val_loss < val_loss_min:
                print(f'Epoch {epoch}: Validation loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                self.save(self.enc_model.state_dict())
                val_loss_min = val_loss
            else:
                print(f'Epoch {epoch}: Validation loss: {val_loss:.6f}')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_loss_min": val_loss_min})

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
            loss = F.l1_loss(pred_embedding, teacher_embedding_mu)
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
        return running_loss / len(self.train_loader)

    def eval(self, the_loader):
        self.enc_model.eval()
        running_loss = 0.0
        all_embeddings = list()
        with torch.no_grad():
            for (data, xrd) in the_loader:
                data = data.to(self.device)
                xrd = xrd.to(self.device).unsqueeze(1)
                pred_embedding = self.enc_model(xrd)
                all_embeddings.append(pred_embedding)
                teacher_embedding_mu, _, _ = self.teacher_model.encode(data)
                loss = F.l1_loss(pred_embedding, teacher_embedding_mu)
                running_loss += loss.item()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        assert all_embeddings.shape == (len(the_loader.dataset), 256)
        return running_loss / len(the_loader), all_embeddings
    
    def save(self, state_dict):
        torch.save(state_dict, os.path.join(self.model_path, f'{self.save_file}_encoder.pt'))


def main():
    # Training settings
    models = {
        'encoder': XRDEncoderTrainer,
    }
    parser = argparse.ArgumentParser(description='Train an XRD module (regressor or encoder)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='number of epochs to train (default: 10)'
    )   
    parser.add_argument(
        '--model_type',
        choices=['encoder'],
        default='encoder',
        help='type of model to train (encoder or regressor)'
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
        default=1e-4,
        metavar='LR',
        help='learning rate'
    )
    args = parser.parse_args()
    args = args.__dict__

    run = wandb.init(
        # Set the project where this run will be logged
        project="xrd_latent_regression",
        config=args
    )

    trainer = models[args.pop('model_type')](**args)
    trainer.train()

if __name__ == "__main__":
    main()