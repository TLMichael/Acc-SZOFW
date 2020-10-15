import torch
from tqdm import tqdm
from torch.utils.data import Dataset

import sys
sys.path.append('../')

from query_model import QueryModel
from utils import load_default, DataFetcher, batch_compute_loss, LMO_L1


DEFAULT_CONFIGURATION = {
    'phishing': {
        'T': 3000, 'theta': 10, 'base_lr': 1, 'log_step': 100
    },
    'a9a': {
        'T': 600, 'theta': 10, 'base_lr': 1, 'log_step': 100
    },
    'w8a': {
        'T': 600, 'theta': 10, 'base_lr': 1, 'log_step': 100
    },
    'covtype': {
        'T': 5500, 'theta': 10, 'base_lr': 1, 'log_step': 400
    },
}


class ZSCG(object):
    """ The ZSCG algorithm """
    def __init__(self, dataset, train_data: Dataset, test_data: Dataset, model: QueryModel, T=1000000, theta=1, base_lr=1, log_step=10):
        """
        Args:
            dataset (str) : the dataset name for loading default configuration.
            train_data (Dataset) : dataset for training.
            test_data (Dataset) : dataset for testing.
            model (QueryModel) : the model for loss computing, gradient estimating and query counting.
            T (int) : the total iteration number for optimization.
            theta (float) : the sparse budget.
            base_lr (float) : the scaling factor of step size.
            log_step (float) : the interval of output information.
        """
        self.dataset = dataset
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.T = T
        self.theta = theta
        self.base_lr = base_lr
        self.log_step = log_step

        load_default(self, DEFAULT_CONFIGURATION)

        self.lr = lambda t: self.base_lr / (T)**(0.75)
        self.delta = lambda t: 2 * model.q**0.5 / model.d**1.5 / (t + 8)**(1/3)
        
        self.batch_size = 100

    def attack(self, writer=None, flag=['TrainLoss', 'TestLoss']):
        with torch.no_grad():
            batch_fetcher = DataFetcher(self.train_data, batch_size=self.batch_size)

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

                    desc = 'Iter {} | TrainLoss {:.6f}  | TestLoss {:.6f} | Norm {:.4f} | lr {:.6f} | delta {:.6f} |'.format(iteration, train_loss, test_loss, weight_norm.item(), self.lr(iteration), self.delta(iteration))
                    tqdm.write(desc)
                    if writer is not None:
                        writer.add_scalar(flag[0], train_loss, global_step=iteration)
                        writer.add_scalar(flag[1], test_loss, global_step=iteration)

                x, y = batch_fetcher.fetch()
                grad = self.model.est_grad(x, y, self.delta(iteration))

                v = LMO_L1(grad, self.theta)
                d = v - self.model.w
                self.model.w = self.model.w + self.lr(iteration) * d

        return train_loss, test_loss

