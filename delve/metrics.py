from typing import Dict, Union
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
COVARIANCE_MATRICES = dict()


def compute_intrinsic_dimensionality(cov: torch.Tensor, thresh: float = 0.99) -> int:
    """
    Compute the intrinsic dimensionality based on the covariance matrix
    :param cov: the covariance matrix as a torch tensor
    :param thresh: delta value; the explained variance of the covariance matrix
    :return: The intrinsic dimensionality; an integer value greater than zero
    """
    eig_vals, eigen_space = cov.symeig(True)
    eig_vals, idx = eig_vals.sort(descending=True)
    eig_vals[eig_vals < 0] = 0
    percentages = eig_vals.cumsum(0) / eig_vals.sum()
    eigen_space = eigen_space[:, percentages < thresh]
    if eigen_space.shape[1] == 0:
        eigen_space = eigen_space[:, :1]
    elif thresh - (percentages[percentages < thresh][-1]) > 0.02:
        eigen_space = eigen_space[:, : eigen_space.shape[1] + 1]
    return eigen_space.shape[1]


def compute_saturation(cov: torch.Tensor, thresh: float = 0.99) -> float:
    """
    Computes the saturation
    :param cov: the covariance matrix as a torch tensor
    :param thresh: delta value; the explained variance of the covariance matrix
    :return: a value between 0 and 1
    """
    intrinsic_dimensionality = compute_intrinsic_dimensionality(cov, thresh)
    feature_space_dimensionality = cov.shape[0]

    return intrinsic_dimensionality / feature_space_dimensionality


def compute_cov_determinant(cov: torch.Tensor) -> float:
    """
    Computes the determinant of the covariance matrix (also known as generalized variance)
    :param cov: the covariannce matrix as torch tensor
    :return: the determinant
    """
    return cov.det().unsqueeze(dim=0).cpu().numpy()[0]


def batch_mean(batch: np.ndarray):
    """Get mean of first vector in `batch`."""  # TODO: Add support for non-dense layers.
    return np.mean(batch[0])


def batch_cov(batch: np.ndarray):
    """Get covariance of first instance in `batch`."""  # TODO: Add support for non-dense layers.
    return np.cov(batch[0])


def compute_saturation(cov: np.ndarray, thresh: float = 0.99) -> float:
    """
    Computes the trace of the covariance matrix diagonal matrix
    :param cov: the covariannce matrix as torch tensor
    :return: the trace
    """

    eig_vals = np.linalg.eigvalsh(cov)

    # Sort the eigenvalues from high to low
    eig_vals = sorted(eig_vals, reverse=True)
    total_dim = len(cov)
    nr_eigs = get_explained_variance(
        eig_vals=eig_vals, threshold=thresh, return_cum=False
    )
    return get_layer_saturation(nr_eig_vals=nr_eigs, layer_width=total_dim)


class MINE(nn.Module):
    def __init__(self, input_size, layer_size, hidden_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(layer_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        h2 = self.fc3(h1)
        return h2


def compute_mutual_information(mine_model, x_sample, y_sample, epochs):

    mine_optimizer = torch.optim.Adam(mine_model.parameters(), lr=0.01)
    mine_model.to(device)

    for epoch in range(epochs):

        y_shuffle = np.random.permutation(y_sample.detach().cpu().numpy())

        x_sample = Variable(x_sample, requires_grad=True)
        y_sample = Variable(y_sample, requires_grad=True)
        y_shuffle = Variable(
            torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad=True
        )

        x_sample = x_sample.to(device)
        y_sample = y_sample.to(device)
        y_shuffle = y_shuffle.to(device)

        pred_xy = mine_model(x_sample, y_sample)
        pred_x_y = mine_model(x_sample, y_shuffle)

        ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss_ = -ret  # maximize
        mine_model.zero_grad()
        loss_.backward()
        mine_optimizer.step()
    return ret
