import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import sys
sys.path.append('../')

from query_model import QueryModel
from utils import load_default, DataFetcher, batch_est_grad, batch_compute_loss, LMO_L1


DEFAULT_CONFIGURATION = {
    'phishing': {
        'T': 3000, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 1, 'log_step': 100
    },
    'a9a': {
        'T': 600, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 1, 'log_step': 100
    },
    'w8a': {
        'T': 600, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 1, 'log_step': 100
    },
    'covtype': {
        'T': 5500, 'theta': 10, 'base_alpha': 1, 'base_eta': 1, 'base_gamma': 1, 'log_step': 400
    },
}


class AccSZOFW(object):
    """ The Acc-SZOFW algorithm """
    def __init__(self, dataset, train_data: Dataset, test_data: Dataset, model: QueryModel, T=1000000, theta=1, base_alpha=1, base_eta=1, base_gamma=1, log_step=10, is_vr=True, is_gm=True, is_vm=True):
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
            is_vr (bool) : variance reduction or not.
            is_gm (bool) : gradient momentum or not.
            is_vm (bool) : variable momentum or not.
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
        self.is_vr, self.is_gm, self.is_vm = is_vr, is_gm, is_vm

        load_default(self, DEFAULT_CONFIGURATION)

        self.alpha = lambda t: self.base_alpha / (1 + t)
        self.eta = lambda t: self.base_eta / (T)**0.5
        self.gamma = lambda t: self.base_gamma * (1 + (1 / (t+1) / (t+2))) * self.eta(t)
        self.gamma2 = lambda t: self.gamma(t) if self.gamma(t) < 1 else 1
        self.mu = 1 / (model.d**0.5 * T**0.5)
        self.beta = 1 / (model.d * T**0.5)
        self.delta = self.mu if model.estimator == 'CooGE' else self.beta

        self.batch_size_1 = min(10000, len(self.train_data))
        self.batch_size_2 = 100
        self.q = 100

        self.little_batch = 2500

        self.beta = lambda t: 4 / (1 + model.d/model.q)**(1/3) / (t + 8)**(2/3)

    def attack(self, writer=None, flag=['TrainLoss', 'TestLoss']):
        with torch.no_grad():
            batch_fetcher_1 = DataFetcher(self.train_data, batch_size=self.batch_size_1)
            batch_fetcher_2 = DataFetcher(self.train_data, batch_size=self.batch_size_2)

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
                
                if self.is_vr:
                    if iteration % self.q == 0:
                        x, y = batch_fetcher_1.fetch()
                        if self.model.d > 200:
                            # batch computing when num_features > 200
                            grad = batch_est_grad(self.model.est_grad, x, y, self.delta, self.little_batch)
                        else:
                            grad = self.model.est_grad(x, y, self.delta)
                    else:
                        x, y = batch_fetcher_2.fetch()
                        grad_new, grad_old = self.model.est_grad(x, y, self.delta, w_old)
                        grad = grad_new - grad_old + grad
                else:
                    x, y = batch_fetcher_2.fetch()
                    grad = self.model.est_grad(x, y, self.delta)
                
                if self.is_gm:
                    # momentum technique
                    if iteration == 0:
                        m = grad
                    m = (1 - self.beta(iteration)) * m + self.beta(iteration) * grad
                    grad = m

                if self.is_vm:
                    v = LMO_L1(grad, self.theta)
                    a = a + self.gamma2(iteration) * (v - a)
                    b = self.model.w + self.eta(iteration) * (v - self.model.w)
                    w_old = self.model.w.clone()
                    self.model.w = (1 - self.alpha(iteration)) * b + self.alpha(iteration) * a
                else:
                    v = LMO_L1(grad, self.theta)
                    w_old = self.model.w.clone()
                    self.model.w = self.model.w + self.lr(iteration) * (v - self.model.w)

        return train_loss, test_loss

