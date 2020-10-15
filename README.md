# Accelerated Stochastic Gradient-free and Projection-free Methods

PyTorch Code for "Accelerated Stochastic Gradient-free and Projection-free Methods".

## Prerequisites

- Python 3.7
- PyTorch 1.3.0
- tensorflow 1.15.2
- tqdm
- pandas
- Pillow
- scikit-learn

## Install

```bash
unzip Acc-SZOFW-main.zip
cd Acc-SZOFW-main
pip install -r requirements.txt
```

## Usage

To solve the robust binary classification problem:
```bash
# Download dataset manually
mkdir -p ~/datasets/phishing/
mkdir -p ~/datasets/a9a/
mkdir -p ~/datasets/w8a/
mkdir -p ~/datasets/covtype/
wget -P ~/datasets/phishing/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
wget -P ~/datasets/a9a/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
wget -P ~/datasets/w8a/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
wget -P ~/datasets/covtype/ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2

cd app2
python run.py
tensorboard --logdir ./results/tsdata/phishing/     # For loss visualization
tensorboard --logdir ./results/tsdata/a9a/      # For loss visualization
tensorboard --logdir ./results/tsdata/w8a/      # For loss visualization
tensorboard --logdir ./results/tsdata/covtype/      # For loss visualization
```



If you find this work useful in your research, please cite using the following BibTeX:
    
    @inproceedings{huang2020accelerated,
        author = {Huang, Feihu and Tao, Lue and Chen, Songcan},
        title = {Accelerated Stochastic Gradient-free and Projection-free Methods},
        booktitle = {International Conference on Machine Learning (ICML)},
        year = {2020}
    }