import numpy as np
import os.path as osp
from sklearn.datasets import load_svmlight_file
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets


# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')

PHISHING_PATH = '~/datasets/phishing/phishing'
A9A_PATH = '~/datasets/a9a/a9a'
W8A_PATH = '~/datasets/w8a/w8a'
COVTYPE_PATH = '~/datasets/covtype/covtype.libsvm.binary.scale.bz2'


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    np.random.seed(0)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class Phishing(Dataset):
    """ `Phishing <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#phishing>`_ Dataset. """
    def __init__(self, path=PHISHING_PATH, train=True):
        self.path = path
        self.split = 'Train' if train else 'Test'
        data = load_svmlight_file(osp.expanduser(self.path))
        X, y = data[0].toarray(), data[1]
        X, y = unison_shuffled_copies(X, y)
        y[y == 0] = -1
        if train:
            X, y = X[:len(y)//2], y[:len(y)//2]
        else:
            X, y = X[len(y)//2:], y[len(y)//2:]
        self.data = X
        self.targets = y
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)
        return x, y

    def __repr__(self):
        head = self.__class__.__name__ + ' ' + self.split
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.path is not None:
            body.append("File location: {}".format(self.path))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)



class A9A(Dataset):
    """ `A9A <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a>`_ Dataset. """
    def __init__(self, path=A9A_PATH, train=True):
        self.path = path
        self.split = 'Train' if train else 'Test'
        data = load_svmlight_file(osp.expanduser(self.path))
        X, y = data[0].toarray(), data[1]
        X, y = unison_shuffled_copies(X, y)
        if train:
            X, y = X[:len(y)//2], y[:len(y)//2]
        else:
            X, y = X[len(y)//2:], y[len(y)//2:]
        self.data = X
        self.targets = y
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)
        return x, y

    def __repr__(self):
        head = self.__class__.__name__ + ' ' + self.split
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.path is not None:
            body.append("File location: {}".format(self.path))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


class W8A(Dataset):
    """ `W8A <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w8a>`_ Dataset. """
    def __init__(self, path=W8A_PATH, train=True):
        self.path = path
        self.split = 'train' if train else 'test'
        data = load_svmlight_file(osp.expanduser(self.path))
        X, y = data[0].toarray(), data[1]
        X, y = unison_shuffled_copies(X, y)
        if train:
            X, y = X[:len(y)//2], y[:len(y)//2]
        else:
            X, y = X[len(y)//2:], y[len(y)//2:]
        self.data = X
        self.targets = y
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)
        return x, y

    def __repr__(self):
        head = self.__class__.__name__ + ' ' + self.split
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.path is not None:
            body.append("File location: {}".format(self.path))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


class CovtypeBinary(Dataset):
    """ `Covtype.binary <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary>`_ Dataset. """
    def __init__(self, path=COVTYPE_PATH, train=True):
        self.path = path
        self.split = 'train' if train else 'test'
        data = load_svmlight_file(osp.expanduser(self.path))
        X, y = data[0].toarray(), data[1]
        X, y = unison_shuffled_copies(X, y)
        y[ y== 2] = -1
        if train:
            X, y = X[:len(y)//2], y[:len(y)//2]
        else:
            X, y = X[len(y)//2:], y[len(y)//2:]
        self.data = X
        self.targets = y
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x, device=DEVICE)
        y = torch.tensor(y, device=DEVICE)
        return x, y

    def __repr__(self):
        head = self.__class__.__name__ + ' ' + self.split
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.path is not None:
            body.append("File location: {}".format(self.path))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


def get_dataset(dataset, train=True):
    if dataset == 'phishing':
        data = Phishing(train=train)
    elif dataset == 'a9a':
        data = A9A(train=train)
    elif dataset == 'w8a':
        data = W8A(train=train)
    elif dataset == 'covtype':
        data = CovtypeBinary(train=train)
    else:
        raise Exception('Unsupported dataset ({}) !'.format(dataset))
    return data


if __name__ == '__main__':
    def count(x, v):
        return (x == v).sum()

    
    # data = Phishing()
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()
    # data = Phishing(train=False)
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()

    # data = A9A()
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()
    # data = A9A(train=False)
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()
    
    # data = W8A()
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()
    # data = W8A(train=False)
    # print(data)
    # print(count(data.targets, 1), count(data.targets, -1))
    # print()
    
    data = CovtypeBinary()
    print(data)
    print(count(data.targets, 1), count(data.targets, -1))
    print()
    data = CovtypeBinary(train=False)
    print(data)
    print(count(data.targets, 1), count(data.targets, -1))
    print()

    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=2)

    
    print('Done')
