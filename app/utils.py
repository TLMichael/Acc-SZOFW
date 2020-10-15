import torch
from torch.utils.data import DataLoader


def load_default(obj, CONFIGURATION):
    option = CONFIGURATION[obj.dataset]
    for k, v in option.items():
        setattr(obj, k, v)


class DataFetcher(object):
    """ Infinite data loader. """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)
        self.dataiter = iter(self.dataloader)
        self.iteration = 0
    
    def fetch(self):
        try:
            x, y = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            x, y = next(self.dataiter)
        self.iteration += 1
        return x, y


def batch_est_grad(func, x, y, delta, batch=100, w_old=None):
    # In avoid of out of memory
    v, v_old = 0, 0
    for idx in range(x.size(0)//batch):
        if w_old is not None:
            tmp, tmp_old = func(x[idx*batch:(idx+1)*batch], y[idx*batch:(idx+1)*batch], delta, w_old)
            v = v + tmp*batch
            v_old = v_old + tmp*batch
        else:
            tmp = func(x[idx*batch:(idx+1)*batch], y[idx*batch:(idx+1)*batch], delta)
            v = v + tmp*batch
    if w_old is not None:
        return v / x.size(0), v_old / x.size(0)
    else:
        return v / x.size(0)


def batch_compute_loss(func, data, batch_size=1000000):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    x, y = next(iter(loader))
    loss = func(x, y)
    return loss.mean().item()

    # loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    # loss = 0
    # for x, y in loader:
    #     batch_loss = func(x, y)
    #     loss += batch_loss.sum().item()
    # loss /= len(data)
    # return loss


def LMO_L1(grad, theta):
    coord = torch.argmax(grad.abs())
    v = torch.zeros_like(grad)
    v[coord] = theta * torch.sign(grad[coord])
    return -v


def P_l1(data, kappa):
    shape = data.shape
    data = data.reshape(-1)
    proj = euclidean_proj_l1ball(data, kappa)
    return proj.reshape(*shape)


""" Module to compute projections on the positive simplex or the L1-ball
A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }
The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }
Adrien Gaidon - INRIA - 2011
source: (https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246)
"""
import numpy as np


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / rho
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w
