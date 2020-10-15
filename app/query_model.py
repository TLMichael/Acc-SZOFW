import torch


# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')


class QueryModel(object):
    """ Robust Lasso Regression Model for loss computing, gradient estimating and query counting. """
    def __init__(self, num_features, q, estimator, sigma=10):
        """
        Args:
            num_features (int) : number of features.
            q (int) : number of samples used for gradient estimator,
                only effective when estimator is GauGE or UniGE.
                -1 means q will be the same as num_features.
            estimator (str) : gradient estimator: ``'GauGE'`` | ``'UniGE'`` | ``'CooGE'``.
            sigma (float) : temperature pamameter of the robust loss.
        """
        self.d = num_features
        self.q = q
        self.query_count = 0
        self.sigma = sigma

        self.estimator = estimator
        assert estimator in ['GauGE', 'UniGE', 'CooGE']

        # Initialize optimization variables with shape [D].
        torch.manual_seed(0)
        # self.w = torch.zeros(self.d, dtype=torch.float64, device=DEVICE)
        self.w = torch.randn(self.d, dtype=torch.float64, device=DEVICE) / 1000
        self.w.requires_grad = True
    
    def compute_loss(self, x, y, w=None):
        """
        Args:
            x (torch.Tensor) : batch data with shape [B x D].
            y (torch.Tensor) : batch label with shape [B].
            w (torch.Tensor) : batch w with shape [B x Q x D].
        """
        c = self.sigma
        if w is None:
            # Batch loss computing for loss
            w = self.w      # (D)
            res = -(y - x @ w)**2
            res = c * c / 2 * (1 - torch.exp(res / (c * c)))
            return res
        else:
            # Batch loss computing for gradient estimating
            pred = torch.bmm(w, x.unsqueeze(dim=2)).squeeze()       # (B, 2Q)
            res = -(y - pred)**2                                    # (B, 2Q)
            res = c * c / 2 * (1 - torch.exp(res / (c * c)))        # (B, 2Q)
            return res
        # return 1 / (1 + torch.exp(y * (x @ w)))
        # return (y - x @ w)**2
        return res
    
    def UniGE(self, x, y, mu, noise=None, w=None):
        with torch.no_grad():
            B = x.size(0)
            Q = self.q
            
            w = self.w if w is None else w      # (D)
            w_left = w.unsqueeze(dim=0).unsqueeze(dim=0).repeat(B, Q, 1)        # (B, Q, D)
            if noise is None:
                noise = torch.rand_like(w_left) * 2 - 1     # Scaling uniform distribution from [0, 1) to [-1, 1)
                norm = noise.view(B*Q, -1).norm(dim=-1).view(B, Q, 1)        # (B, Q, 1)
                noise = noise / norm        # (B, Q, D)

            w_all = torch.cat([w_left, w_left], dim=1)      # (B, 2Q, D)
            noise_all = torch.cat([noise, torch.zeros_like(w_left)], dim=1)

            target = y.unsqueeze(dim=1).repeat(1, 2*Q)      # (B, 2Q)

            loss = self.compute_loss(x, target, w_all + mu * noise_all)     # (B, 2Q)
            loss_left, loss_right = loss[:, :Q], loss[:, Q:]        # (B, Q), (B, Q)

            grad = (loss_left - loss_right).view(B, Q, 1) * noise   # (B, Q, D)
            grad = torch.mean(grad, dim=1)      # (B, D)
            grad = torch.mean(grad, dim=0)      # (D)
            grad = grad * (self.d / mu)
        return grad, 2 * Q * B, noise
    
    def GauGE(self, x, y, mu, noise=None, w=None):
        with torch.no_grad():
            B = x.size(0)
            Q = self.q

            w = self.w if w is None else w      # (D)
            w_left = w.unsqueeze(dim=0).unsqueeze(dim=0).repeat(B, Q, 1)        # (B, Q, D)
            if noise is None:
                noise = torch.randn_like(w_left)    # Normal distribution with mean 0 and variance 1

            w_all = torch.cat([w_left, w_left], dim=1)      # (B, 2Q, D)
            noise_all = torch.cat([noise, torch.zeros_like(w_left)], dim=1)     # (B, 2Q, D)

            target = y.unsqueeze(dim=1).repeat(1, 2*Q)      # (B, 2Q)

            loss = self.compute_loss(x, target, w_all + mu * noise_all)     # (B, 2Q)
            loss_left, loss_right = loss[:, :Q], loss[:, Q:]        # (B, Q), (B, Q)

            grad = (loss_left - loss_right).view(B, Q, 1) * noise   # (B, Q, D)
            grad = torch.mean(grad, dim=1)      # (B, D)
            grad = torch.mean(grad, dim=0)      # (D)
            grad = grad / mu
        return grad, 2 * Q * B, noise
    
    def CooGE(self, x, y, mu, w=None):
        with torch.no_grad():
            B = x.size(0)
            D = self.d

            w = self.w if w is None else w      # (D)
            w_left = w.unsqueeze(dim=0).unsqueeze(dim=0).repeat(B, D, 1)        # (B, D, D)

            # Generate standard basis tensor
            basis_set = torch.eye(D, D).to(DEVICE)
            basis_left = basis_set.unsqueeze(dim=0).repeat(B, 1, 1)     # (B, D, D)

            w_all = torch.cat([w_left, w_left], dim=1)      # (B, 2D, D)
            noise_all = torch.cat([basis_left, -basis_left], dim=1)     # (B, 2D, D)
            
            target = y.unsqueeze(dim=1).repeat(1, 2*D)      # (B, 2D)

            loss = self.compute_loss(x, target, w_all + mu * noise_all)     # (B, 2D)
            loss_left, loss_right = loss[:, :D], loss[:, D:]        # (B, D), (B, D)

            grad = (loss_left - loss_right).view(B, D, 1) * basis_left      # (B, D, D)
            grad = torch.sum(grad, dim=1)      # (B, D)
            grad = torch.mean(grad, dim=0)      # (D)
            grad = grad / (2 * mu)
        return grad, 2 * D * B

    def est_grad(self, x, y, mu, w_old=None):
        # Estimate gradient using GauGE/UniGE/CooGE
        if self.estimator == 'GauGE':
            grad, query_size, noise = self.GauGE(x, y, mu)
            self.query_count += query_size
            if w_old is not None:
                grad_old, query_size, _ = self.GauGE(x, y, mu, noise, w_old)
                self.query_count += query_size
        elif self.estimator == 'UniGE':
            grad, query_size, noise = self.UniGE(x, y, mu)
            self.query_count += query_size
            if w_old is not None:
                grad_old, query_size, _ = self.UniGE(x, y, mu, noise, w_old)
                self.query_count += query_size
        elif self.estimator == 'CooGE':
            grad, query_size = self.CooGE(x, y, mu)
            self.query_count += query_size
            if w_old is not None:
                grad_old, query_size = self.CooGE(x, y, mu, w_old)
                self.query_count += query_size
        if w_old is not None:
            return grad, grad_old
        else:
            return grad
 
    def auto_grad(self, x, y):
        # Compute gradient using autograd
        loss = self.compute_loss(x, y)  # (B)
        loss = torch.mean(loss)     
        grad, = torch.autograd.grad(loss, [self.w])      # (D)
        return grad


if __name__ == '__main__':
    B, D = 3, 100
    Q = 1

    model = QueryModel(D, Q, 'UniGE', 1)

    x = torch.randn(B, D, dtype=torch.float64).to(DEVICE)
    y = torch.randn(B, dtype=torch.float64).to(DEVICE)

    auto_g = model.auto_grad(x, y)

    def print1(method, mu):
        if method == 'uni_grad':
            uni_g, _, _ = model.UniGE(x, y, mu)
        elif method == 'gau_grad':
            uni_g, _, _ = model.GauGE(x, y, mu)
        elif method == 'coo_grad':
            uni_g, _ = model.CooGE(x, y, mu)
        else:
            uni_g = auto_g
        print(method, '{:.7f}'.format(mu), uni_g[0:5].cpu().detach() > 0, ((uni_g > 0) == (auto_g > 0)).sum().item(), uni_g.mean().item())
    
    print1('auto_grad', 0)
    print()

    print1('uni_grad', 1)
    print1('uni_grad', 0.1)
    print1('uni_grad', 0.01)
    print1('uni_grad', 0.001)
    print1('uni_grad', 0.00001)
    print1('uni_grad', 0.0000001)
    print()

    print1('gau_grad', 1)
    print1('gau_grad', 0.1)
    print1('gau_grad', 0.01)
    print1('gau_grad', 0.001)
    print1('gau_grad', 0.00001)
    print1('gau_grad', 0.0000001)
    print()

    print1('coo_grad', 1)
    print1('coo_grad', 0.1)
    print1('coo_grad', 0.01)
    print1('coo_grad', 0.001)
    print1('coo_grad', 0.00001)
    print1('coo_grad', 0.0000001)