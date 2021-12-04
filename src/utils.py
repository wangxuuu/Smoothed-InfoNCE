import torch
import numpy as np
import random
from matplotlib import pyplot as plt

DEVICE = "cuda"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def plot_subfigure(net, X, Y, dimX, dimY, x0=None, y0=None, xmin=-5, xmax=5, ymin=-5, ymax=5, xgrids=50, ygrids=50, ax=None, show_details=True):
    """
    The inputs should be X and Y, which are the coordinates of the points.

    net should be a neural network with Tensor inputs.
    """

    if  x0 == None:
        x0 = np.zeros((1, X.shape[1]))
    if y0 == None:
        y0 = np.zeros((1, Y.shape[1]))
        
    x, y = np.mgrid[xmin:xmax:xgrids * 1j, ymin:ymax:ygrids * 1j]
    with torch.no_grad():
        z = (net(
            torch.cat((torch.Tensor((np.arange(X.shape[-1]) == dimX).reshape(1, -1) *
                    x.reshape(-1, 1) + x0).to(DEVICE),
            torch.Tensor((np.arange(X.shape[-1]) == dimY).reshape(1, -1) *
                    y.reshape(-1, 1) + y0).to(DEVICE),
        ),dim=-1)).reshape(x.shape).cpu())
    if ax is None:
        ax = plt.gca()

    im = ax.pcolormesh(x, y, z, cmap="RdBu_r", shading="auto")
    # ax.figure.colorbar(im)
    if show_details:
        ax.figure.colorbar(im) 
        ax.set(xlabel="$x^{{({0})}}-x_0^{{({0})}}$",
                ylabel="$x^{{({0})}}-x_0^{{({0})}}$",
                title=r"Heatmap of $t(x,y)$")
    return im


def plot_fig(net, X, Y, d=6):
    DEVICE = "cuda"
    f, axs = plt.subplots(nrows=d,ncols=d,sharex=True, sharey=True)
    for i in range(d):
        for j in range(d):
            im = plot_subfigure(net, X, Y, dimX=i, dimY=j, x0=None, y0=None, xmin=-5, xmax=5, ymin=-5, ymax=5, xgrids=50, ygrids=50, ax=axs[i, j], show_details=False)
    f.colorbar(im, ax=axs.ravel().tolist())
    return f