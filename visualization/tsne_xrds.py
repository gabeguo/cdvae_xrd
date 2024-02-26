from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt

X = torch.load('/home/gabeguo/cdvae/data/mp_20/xrd/val.pt').detach().cpu().numpy()
print(X.shape)
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
print(X_embedded.shape)

plt.scatter(X_embedded[:,0], X_embedded[:,1], s=1)
plt.title('t-SNE of 512-dimensional XRD patterns')
plt.savefig('tsne_xrd.png')
plt.close()