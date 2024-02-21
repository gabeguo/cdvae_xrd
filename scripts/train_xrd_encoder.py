
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from cdvae.pl_modules.xrd_encoder import XRDEncoder


class XRDTrainer:
    def __init__(self, data_dir, save_dir, model_path, batch_size, epochs, lr):
        
        self.data_dir = data_dir
        self.save_dir = save_dir
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
        pass

    def load_teacher_model(self):
        pass

    def create_enc_model(self):
        self.enc_model = XRDEncoder()
        self.enc_model.to(self.device)

    def train(self):
        pass

    def train_epoch(self):
        pass
    
    def save(self):
        pass

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train the XRD encoder to predict latent molecule representations')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)'
    )                   
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        metavar='N',
        help='input batch size for training (default: 64)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
    )
    parser.add_argument(
        '--data_dir',
        type=str,
    )
    parser.add_argument(
        '--save_dir',
        type=str,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        metavar='LR',
        help='learning rate (default: 0.01)'
    )
    args = parser.parse_args()
    trainer = XRDTrainer(**vars(args))
    trainer.train()

if __name__ == "__main__":
    main()