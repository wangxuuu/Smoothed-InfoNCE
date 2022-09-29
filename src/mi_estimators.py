# Store many mutual information estimators, including MINE, InfoNCE, MINEE, GAN-MINEE, PCM, AdapLS

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import Tensor


a, b, c = 0.01, 1e-8, 1 - 1e-8


cuda = True if torch.cuda.is_available() else False
DEVICE = "cuda" if cuda else "cpu"

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

def randerange(*args, **kwargs):
    s = torch.randperm(*args, **kwargs)
    t = torch.randperm(*args, **kwargs)
    b = s != t
    return s[b], t[b]

def uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min

def div_reg(net, data, ref):
    """
    Regulize the second term of the loss function
    """
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref - log_mean_ef_ref**2

def div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

def nwj_div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    return net(data).mean() - (net(ref)-1).exp().mean()


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        # self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, 1))
        self.F_func = Net(x_dim + y_dim, hidden_size, sigma=0.02)

    def step(self, x_samples, y_samples, reg=False):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_samples, y_shuffle], dim=-1))
        L = torch.logsumexp(T1, dim=0) - np.log(sample_size)
        if reg:
            loss =  - T0.mean() + L + L**2
        else:
            loss = - T0.mean() + L

        return loss

    def mi_est(self, x_samples, y_samples, writer=None, epoch=None):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_samples, y_shuffle], dim=-1))

        a = T0.mean()
        b = torch.logsumexp(T1, dim=0) - np.log(sample_size)

        if writer is not None:
            writer.add_scalar('MINE/a', a, epoch)
            writer.add_scalar('MINE/b', b, epoch)

        return a - b
        lower_bound = a-b

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        # self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, 1))
        self.F_func = Net(x_dim + y_dim, hidden_size, sigma=0.02)

    def mi_est(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        # self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, 1),
        #                             nn.Softplus())
        self.F_func = Net(x_dim + y_dim, hidden_size, sigma=0.02)

    def mi_est(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [sample_size, sample_size, 1]
        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound


class PCM(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(PCM, self).__init__()
        self.F_func = Prob_Net(x_dim + y_dim, hidden_size, sigma=0.02)

    def forward(self, x_samples, y_samples):
        xy_samples = torch.cat((x_samples, y_samples), dim=1)
        sample_size = y_samples.shape[0]
        X_ref = resample(x_samples, batch_size=sample_size)
        Y_ref = resample(y_samples, batch_size=sample_size)

        data_margin = torch.cat((X_ref, Y_ref), dim=1)
        valid = Variable(torch.Tensor(sample_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.Tensor(sample_size, 1).fill_(0.0), requires_grad=False)
        train_data = torch.cat((xy_samples, data_margin), dim=0)
        labels = torch.cat((valid, fake), dim=0)
        pred_label = self.F_func(train_data)
        # alpha = data_margin.shape[0] / xy_samples.shape[0]

        loss = torch.nn.BCELoss()(pred_label, labels)
        return loss

    def mi_est(self, x_samples, y_samples, gamma=1e-4):
        xy_samples = torch.cat((x_samples, y_samples), dim=1)
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        data_margin = torch.cat((x_tile, y_tile), dim=1)

        alpha = data_margin.shape[0] / xy_samples.shape[0]

        mi = mi_estimate(self.F_func, xy_samples, gamma, alpha)
        return mi


class AdapLS(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(AdapLS, self).__init__()
        self.F_func = AdapLSNet(x_dim + y_dim, hidden_size, sigma=0.02)

    def forward(self, x_samples, y_samples):
        xy_samples = torch.cat((x_samples, y_samples), dim=1)
        sample_size = y_samples.shape[0]
        X_ref = resample(x_samples, batch_size=sample_size)
        Y_ref = resample(y_samples, batch_size=sample_size)

        data_margin = torch.cat((X_ref, Y_ref), dim=1)
        valid = Variable(torch.Tensor(sample_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.Tensor(sample_size, 1).fill_(0.0), requires_grad=False)
        train_data = torch.cat((xy_samples, data_margin), dim=0)
        labels = torch.cat((valid, fake), dim=0)
        pred_label, alpha = self.F_func(train_data)
        # c_0_1_ratio = data_margin.shape[0] / xy_samples.shape[0]

        loss = smooth_ce_loss(pred_label, labels, alpha.detach(), 2)
        return loss

    def mi_est(self, x_samples, y_samples, gamma=0):
        xy_samples = torch.cat((x_samples, y_samples), dim=1)
        sample_size = y_samples.shape[0]
        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))
        data_margin = torch.cat((x_tile, y_tile), dim=1)

        c_0_1_ratio = data_margin.shape[0] / xy_samples.shape[0]

        mi = mi_estimate_adap(self.F_func, xy_samples, gamma, c_0_1_ratio)
        return mi

    # def add_record(self, writer, loss, ):
    #     writer.add_scalar('loss/train', loss, w_idx)
    #     #writer.add_scalar('acc/train', acc_train, i)
    #     writer.add_scalar("normalized mi_estimate/mi_est divided by ground_truth", mi_est.item()/Ground_truth, w_idx)
    #     writer.add_histogram('labels/predicted', pred_labels, w_idx, bins=np.arange(1000)/999)
    #     writer.add_histogram('alpha', alpha, w_idx)


# def log_sum_exp(value, dim=None, keepdim=False):
#     """Numerically stable implementation of the operation
#     value.exp().sum(dim, keepdim).log()
#     """
#     if dim is not None:
#         m, _ = torch.max(value, dim=dim, keepdim=True)
#         value0 = value - m
#         if keepdim is False:
#             m = m.squeeze(dim)
#         return m + torch.log(torch.sum(torch.exp(value0),
#                                        dim=dim, keepdim=keepdim))
#     else:
#         m = torch.max(value)
#         sum_exp = torch.sum(torch.exp(value - m))
#         return m + torch.log(sum_exp)

def acti_func(x, a, b, c):
    # a is \alpha_0, b is \tau and c is 1-\tau in the paper
    alpha = torch.zeros_like(x)
    x_cpu = x.cpu()
    alpha[np.where(x_cpu.cpu() <= b)] = - a * x[np.where(x_cpu <= b)] / b + a
    alpha[np.where((x_cpu > b) & (x_cpu < c))] = 0
    alpha[np.where(x_cpu >= c)] = a * x[np.where(x_cpu >= c)] / (1 - c) + a * c / (c - 1)
    return alpha


def mi_estimate(model, test_XY, gamma, alpha):
    # clip the output of neural networks
    pre = model(test_XY).clamp(min=gamma, max=1 - gamma)
    MI_est = torch.log(alpha * pre / ((1 - pre).clamp(min=gamma, max=1 - gamma))).mean()

    return MI_est


def mi_estimate_adap(model, XY, gamma, c_0_1_ratio):
    # c_0_1_ratio = p_{C}(0) / p_{C}(1)
    pre, _ = model(XY)
    pre = pre.clamp(min=gamma, max=1 - gamma)  # clip the output pre controled by gamma; gamma = 0 means no clipping
    MI_est = torch.log(c_0_1_ratio * pre / ((1 - pre).clamp(min=gamma, max=1 - gamma))).mean()

    return MI_est


def smooth_ce_loss(pre_label, true_label, smoothing, num_classes):
    new_labels = (1.0 - smoothing) * true_label + smoothing / num_classes
    return torch.nn.BCELoss()(pre_label, new_labels)


class Net(nn.Module):
    # Inner class that defines the neural network architecture
    # The output is ranged in R.
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):
        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
        output = self.fc[self.hidden_layers](output)
        return output


class Prob_Net(nn.Module):
    # Inner class that defines the neural network architecture
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):

        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
        output = torch.sigmoid(self.fc[self.hidden_layers](output))
        return output


class AdapLSNet(nn.Module):
    # Inner class that defines the neural network architecture
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):
        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
        output = torch.sigmoid(self.fc[self.hidden_layers](output))

        alpha = acti_func(output, a, b, c)

        return output, alpha


class Generator(nn.Module):
    def __init__(self, input_dim=2, y_dim=2, hidden_size=100, sigma=0.02):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, y_dim)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, noise):
        gen_input = noise
        output = F.elu(self.fc1(gen_input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

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
    # im = ax.pcolormesh(x, y, z, cmap="RdBu_r", shading="auto")
    im = ax.pcolormesh(x, y, z, cmap="RdBu_r", shading="auto")
    # ax.figure.colorbar(im)
    if show_details:
        ax.figure.colorbar(im) 
        ax.set(xlabel="$x^{{({0})}}-x_0^{{({0})}}$",
                ylabel="$x^{{({0})}}-x_0^{{({0})}}$",
                title=r"Heatmap of $t(x,y)$")
    return im


def shuffle_data(X, Y, sample_size, exc=False):
    """
    Shuffle the data.

    sample_size: the number of samples to be shuffled.

    If exc is True, the data will be shuffled without the original data.
    If exc is False, the data will be shuffled with the original data.
    """
    if exc:
        while True:
            idx1 = torch.randperm(X.shape[0])[:sample_size]
            idx2 = torch.randperm(X.shape[0])[:sample_size]
            if (idx1==idx2).sum() == 0:
                break
        return X[idx1], Y[idx2]
    else:
        ref_X = resample(X, sample_size)
        ref_Y = resample(Y, sample_size)
        return ref_X, ref_Y

class GaussianData():

    def __init__(self,
                 n,
                 d=1,
                 muX=0,
                 muY=0,
                 rho=0.9,
                 prior=1/2,
                 rng=None,
                 seed=None,
                 device=None):
        """
        Construct n i.i.d. samples of jointly gaussian (X, Y) for mutual
        information estimation.

        The model for data is
            X:=(x1, x2, ..., xd)
            Y:=(y1, y2, ..., yd)
        where (xi, yi)'s are i.i.d. jointly gaussian with
            mean: (muX, muY)
            covariance:
                [[1, rho],
                 [rho, 1]]

        Parameters:
        -----------
        n: int
            Number of samples.
        d: int
            Number of elements of X (and Y).
        muX: float
            Mean of each element xi, same for all i.
        muY: float
            Mean of each element yi, same for all i.
        rho: float within (-1, 1)
            Correlation between xi and yi.
        rng: numpy.random._generator.Generator
            For generating the random samples.
        seed:
            If rng is None, it is set to numpy.random.default_rng(seed).
        device:
            Default device to use for samples of X and Y.
        """
        self.d, self.n, self.rho, self.device = d, n, rho, device
        rng = rng or np.random.default_rng(seed)
        self.sampler = lambda *args, **kwargs: rng.multivariate_normal(
            (muX, muY), [[1, rho], [rho, 1]], *args, **kwargs)
        self.resample()

    def to(self, *args, **kwargs):
        """
        Move and/or casts the tensors self.X and self.Y.

        Calls torch.Tensor.to(*args, **kwargs) on self.X and self.Y.
        """
        self._X = self._X.to(*args, **kwargs)
        self._Y = self._Y.to(*args, **kwargs)

    def resample(self):
        """
        Resamples X and Y.
        """
        self._XY = self.sampler(self.n * self.d).reshape(self.n, self.d, 2,
                                               order='F').transpose(0, 2, 1)
        self._X = Tensor(self._XY[:, 0, :])
        self._Y = Tensor(self._XY[:, 1, :])
        if self.device is not None:
            self.to(self.device)
        return self

    @property
    def X(self):
        """
        Tensor: Samples of X.
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        return self._X

    @property
    def Y(self):
        """
        Tensor: Samples of Y.
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        return self._Y

    def mutual_information(self):
        """
        Returns the ground truth mutual information I(X;Y).
        """
        return -0.5 * np.log(1 - self.rho**2) * self.d

    IXY = property(mutual_information)

    def pointwise_mutual_information(self, x, y):
        """
        Returns the log density ration pXY/(pX*pY).

        Parameters:
        -----------
        x: Tensor
            Dimension 0: samples of X
            Dimension 1: features of X
        y: Tensor
            Dimension 0: samples of X
            Dimension 1: features of X
        """
        c = self.rho / (2 * (1 - self.rho**2))
        return self.mutual_information() + c * (2 * x * y - self.rho *
                                                (x**2 + y**2)).sum(dim=1)

    iXY = pointwise_mutual_information


    def mutual_information_std(self, size=1000000):
        """
        Returns the standard deviation of the sample average of pointwise mutual information.

        This is the variation in the estimate attributed solely to the number of samples limited to n, 
        since true log density ratio is used to compute the sample average.
        """
        xy = self.sampler(size)
        x = xy[:, 0]
        y = xy[:, 1]
        delta_squared = (2 * x * y - self.rho * (x**2 + y**2))**2
        m, s = delta_squared.mean(), delta_squared.std(ddof=1)
        c = (self.d / self.n)**0.5 * self.rho / (2 * (1 - self.rho**2))
        print(f'percentage error: +/- {2*s/m/(size)**0.5 * 100:.1g} %')
        return m * c

    IXY_std = property(mutual_information_std)

    def plot(self, dimX=0, dimY=0, ax=None, hide_details=False, **kwargs):
        """
        Scatter plot of X[dimX] and Y[dimY].

        Parameters:
        -----------
        dimX: int
            The dimension of X to plot along the x-axis.
        dimY: int
            The dimension of X to plot along the y-axis.
        ax: axis
            The axis to plot the data
        hide_details: bool
            Add xlabel, ylabel, and title if false.
        """
        if ax is None:
            ax = plt.gca()
        if not hide_details:
            ax.set(xlabel=f'$\\mathsf{{x}}_{{{dimX}}}$',
                   ylabel=f'$\\mathsf{{y}}_{{{dimY}}}$',
                   title=f'Data samples')
        ax.scatter(self._XY[:, 0, dimX], self._XY[:, 1, dimY], **kwargs)