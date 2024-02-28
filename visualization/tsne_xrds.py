from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

X = torch.load('/home/gabeguo/cdvae/data/mp_20/xrd/val.pt').detach().cpu().numpy()
print(X.shape)
X_embedded = TSNE(n_components=2, perplexity=25, learning_rate=10).fit_transform(X)
print(X_embedded.shape)

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_embedded)
labels = kmeans.labels_

# Thanks https://stackoverflow.com/a/32740814

# define the colormap
cmap = plt.cm.rainbow
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,n_clusters,n_clusters+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

plt.scatter(X_embedded[:,0], X_embedded[:,1], s=1, c=labels, cmap=cmap, norm=norm)
plt.title('t-SNE of 512-dimensional XRD patterns')
plt.savefig('tsne_xrd.png')
plt.close()