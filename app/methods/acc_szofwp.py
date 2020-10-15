import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import sys
sys.path.append('../')

from query_model import QueryModel
from utils import load_default, DataFetcher, batch_est_grad, batch_compute_loss, LMO_L1


DEFAULT_CONFIGURATION = {
    'phishing': {
        'T': 3000, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 6, 'log_step': 100
    },
    'a9a': {
        'T': 600, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 6, 'log_step': 100
    },
    'w8a': {
        'T': 600, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 6, 'log_step': 100
    },
    'covtype': {
        'T': 5500, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 6, 'log_step': 400
    },
}


class AccSZOFWP(object):
    """ The Acc-SZOFW* algorithm """
    def __init__(self, dataset, train_data: Dataset, test_data: Dataset, model: QueryModel, T=1000000, theta=1, base_alpha=1, base_eta=1, base_gamma=1, log_step=10):
        """
        Args:
            dataset (str) : the dataset name for loading default configuration.
            train_data (Dataset) : dataset for training.
            test_data (Dataset) : dataset for testing.
            model (QueryModel) : the model for loss computing, gradient estimating and query counting.
            T (int) : the total iteration number for optimization.
            eps (float) : the scaling factor of step size.
            base_alpha (float) : the weighted parameter.
            base_eta (float) : the step size.
            base_gamma (float) : the step size.
            log_step (float) : the interval of output information.
        """
        self.dataset = dataset
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.T = T
        self.theta = theta
        self.base_alpha = base_alpha
        self.base_eta = base_eta
        self.base_gamma = base_gamma
        self.log_step = log_step

        load_default(self, DEFAULT_CONFIGURATION)

        self.alpha = lambda t: self.base_alpha / (1 + t)
        self.eta = lambda t: self.base_eta / (T)**(2/3)
        self.gamma = lambda t: self.base_gamma * (1 + (1 / (t+1) / (t+2))) * self.eta(t)
        self.gamma2 = lambda t: self.gamma(t) if self.gamma(t) < 1 else 1
        self.rho = lambda t: 1 / (1 + t)**(2/3)
        self.mu = 1 / (model.d**0.5 * T**(2/3))
        self.beta = 1 / (model.d * T**(2/3))
        self.delta = self.mu if model.estimator == 'CooGE' else self.beta

        self.batch_size = 100

    def attack(self, writer=None, flag=['TrainLoss', 'TestLoss']):
        with torch.no_grad():
            batch_fetcher = DataFetcher(self.train_data, batch_size=self.batch_size)

            w_old = self.model.w.clone()
            a, b = self.model.w.clone(), self.model.w.clone()
            for iteration in tqdm(range(self.T)):

                if iteration % self.log_step == 0:
                    # Sanity check
                    weight_norm = self.model.w.abs().sum()
                    is_valid = weight_norm <= (self.theta + 1e-6)
                    if not is_valid:
                        print('!!! ??? weight out of range ({} > {})'.format(weight_norm, self.theta))
                        return 0
                    
                    train_loss = batch_compute_loss(self.model.compute_loss, self.train_data)
                    test_loss = batch_compute_loss(self.model.compute_loss, self.test_data)

                    desc = 'Iter {} | TrainLoss {:.6f}  | TestLoss {:.6f} | Norm {:.4f} | gamma {:.6f} | eta {:.6f} | alpha {:.6f} |'.format(iteration, train_loss, test_loss, weight_norm.item(), self.gamma2(iteration), self.eta(iteration), self.alpha(iteration))
                    tqdm.write(desc)
                    if writer is not None:
                        writer.add_scalar(flag[0], train_loss, global_step=iteration)
                        writer.add_scalar(flag[1], test_loss, global_step=iteration)
                
                if iteration == 0:
                    x, y = batch_fetcher.fetch()
                    grad = self.model.est_grad(x, y, self.delta)
                else:
                    x, y = batch_fetcher.fetch()
                    grad_new, grad_old = self.model.est_grad(x, y, self.delta, w_old)
                    grad = grad_new + (1 - self.rho(iteration)) * (grad - grad_old)

                v = LMO_L1(grad, self.theta)
                a = a + self.gamma2(iteration) * (v - a)
                b = self.model.w + self.eta(iteration) * (v - self.model.w)
                w_old = self.model.w.clone()
                self.model.w = (1 - self.alpha(iteration)) * b + self.alpha(iteration) * a

        return train_loss, test_loss

