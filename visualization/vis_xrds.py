from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl

from cdvae.pl_data.dataset import CrystDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from scripts.eval_utils import load_model

import argparse

def tsne(args, X, Z):
    # Define color map
    # Thanks https://stackoverflow.com/a/32740814
    # define the colormap
    cmap = plt.cm.rainbow
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,args.n_clusters,args.n_clusters+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # XRD Part
    X = X.detach().cpu().numpy()
    print(X.shape)
    X_embedded = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=10).fit_transform(X)
    print(X_embedded.shape)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, n_init="auto").fit(X_embedded)
    labels = kmeans.labels_

    plt.scatter(X_embedded[:,0], X_embedded[:,1], s=1, c=labels, cmap=cmap, norm=norm)
    plt.title('t-SNE of 512-dimensional XRD patterns')
    plt.savefig(os.path.join(args.save_dir, 'tsne_xrd.png'))
    plt.close()

    # Compare Latents
    Z = Z.detach().cpu().numpy()
    print(np.isnan(Z).any())
    print(np.isinf(Z).any())
    Z_embedded = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=10).fit_transform(Z)
    print(Z_embedded.shape)

    plt.scatter(Z_embedded[:,0], Z_embedded[:,1], s=1, c=labels, cmap=cmap, norm=norm)
    plt.title('t-SNE of embedded latent codes')
    plt.savefig(os.path.join(args.save_dir, 'tsne_latent.png'))
    plt.close()

    return

def plot_xrds(xrds, output_dir, num_materials, Qs):
    for i in range(min(num_materials, xrds.shape[0])):
        curr_xrd = xrds[i]
        assert curr_xrd.shape == (512,)
        plt.plot(Qs, curr_xrd)
        plt.savefig(os.path.join(output_dir, f'material{i}.png'))
        plt.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE of XRD vs embedding')

    parser.add_argument(
        '--model_path',
        default='/home/gabeguo/hydra/singlerun/2024-03-26/mp_20_sincFilter_size10',
        type=str,
    )
    parser.add_argument(
        '--data_dir',
        default='/home/gabeguo/cdvae_xrd/data/mp_20',
        type=str,
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
    )
    parser.add_argument(
        '--save_dir',
        default='vis_outputs',
        type=str
    )
    parser.add_argument(
        '--n_clusters',
        default=10,
        type=int
    )
    parser.add_argument(
        '--perplexity',
        default=25,
        type=float
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model, data_loader, cfg = load_model(
        Path(args.model_path), load_data=True)

    Z = list()
    X = list()

    for idx, batch in enumerate(data_loader):
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z.detach()
        print('shape before (XRD):', batch.y.shape)
        scaled_xrds = batch.y.reshape(-1, 512)
        print('shape after (XRD):', scaled_xrds.shape)
        # inverse transform
        # model.scaler.match_device(scaled_xrds)
        # xrds = model.scaler.inverse_transform(scaled_xrds)
        assert torch.equal(scaled_xrds[0], batch.y[:512, 0])
        xrds = scaled_xrds

        print('shape before (sinc):', batch.raw_sinc.shape)
        sinc_only = batch.raw_sinc.reshape(-1, 512)
        print('shape after (sinc):', sinc_only.shape)
        assert torch.equal(sinc_only[0], batch.raw_sinc[:512, 0])

        # plot XRDs
        print('plotting XRDs')
        if idx == 0:
            pred_xrds = model.fc_property(z)
            # pred_xrds = model.scaler.inverse_transform(pred_xrds)
            pred_xrds = pred_xrds.detach().cpu().numpy()
            pred_xrd_dir = os.path.join(args.save_dir, 'pred_xrds')
            os.makedirs(pred_xrd_dir, exist_ok=True)
            plot_xrds(pred_xrds, output_dir=pred_xrd_dir, num_materials=10, Qs=data_loader.dataset.Qs)

            gt_xrds = xrds.detach().cpu().numpy()
            gt_xrd_dir = os.path.join(args.save_dir, 'gt_xrds')
            os.makedirs(gt_xrd_dir, exist_ok=True)
            plot_xrds(gt_xrds, output_dir=gt_xrd_dir, num_materials=10, Qs=data_loader.dataset.Qs)

            sinc_only = sinc_only.detach().cpu().numpy()
            sinc_only_dir = os.path.join(args.save_dir, 'sinc_only')
            os.makedirs(sinc_only_dir, exist_ok=True)
            plot_xrds(sinc_only, output_dir=sinc_only_dir, num_materials=10, Qs=data_loader.dataset.Qs)


        # Z.append(z)
        # X.append(xrds)
    
    # Z = torch.cat(Z, dim=0)
    # X = torch.cat(X, dim=0)

    # assert Z.shape == (X.shape[0], 256)
    # assert X.shape == (Z.shape[0], 512)

    # tsne(args, X=X, Z=Z)
