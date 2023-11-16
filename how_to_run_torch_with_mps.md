# 如何在 Mac(Apple Silicon) 安装可以调用 mps 的 torch

corresponding link:[英文版本](https://github.com/jeffheaton/app_deep_learning/blob/main/install/pytorch-install-aug-2023.ipynb) 



## 愿景

本教程旨在教会您如何在 Mac(apple Silicon)上安装 PyTorch 并运行一个简单的神经网络来测试性能。

## 动机

目前，基于Apple silicon 的 Mac 已经在大语言模型推理上表现了极高的性价比(192GB)运存的 Mac studio 只需要 4W+  人民币，而 80G*2 A100 加上整个系统的搭建可能花费要超过 20W 人民币，相较于 NVIDIA 对中国境内的各种不可抗力的禁售，Mac 的断购可能性比较小。在这里简单的进行 Mac 如何在 Mac(Apple Silicon)上面的安装和运行进行探索，同时我想要做一些高度量化的LLM然后直接在 Mac 上运行的 App，所以，这是第一步。

## 安装 Python 和 Pytorch

在云时代，大家可能会选择在 Google Colab 上进行自己的小模型的部署以及推理（毕竟免费），但是其实在自己的本地的电脑上进行深度学习的搭建是完全可能的。我们所需要的就是对应着官网要求的硬件进行购买以及硬件与软件的关系进行环境适配。Windows+cuda(次主流)，Linux+cuda(最主流)的相关教程已经非常多了，在这里我们仅对Mac(Apple silicon)如何安装Pytorch 环境进行介绍，参考链接为[英文版本](https://github.com/jeffheaton/app_deep_learning/blob/main/install/pytorch-install-aug-2023.ipynb) 。

1. 我们需要做的第一步是安装 Python 3.9。我强烈推荐使用 Miniconda，它是广博的 Anaconda Python版本中最精简的一部分，其实很多时候我们并不需要那么多无用的包，适合自己的研究需求就好。同时因为我们是 Mac(Apple silicon)，我们理应最大化 M1 MPS 的效率，享用最先进的芯片设计，内存架构简并化带来的训练优势，并不断的优化这个架构下的各种潜在 bug。 所以请按照我们的OS 和硬件在miniconda 的官网上下载最适合和最稳健的 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 版本。

   截止到我写这个 post 之前(2023.11.12 22.55pm)，我是下面的方式来安装 miniconda3 的：

   ```bash
   mkdir -p ~/miniconda3
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm -rf ~/miniconda3/miniconda.sh
   ```

   在安装后，使用刚刚安装的 Miniconda 来初始进行初始化，因为我使用了 zsh，所以多了一行，只用默认终端的同学直接初始化第一行就行：

   ```bash
   ~/miniconda3/bin/conda init bash
   ~/miniconda3/bin/conda init zsh
   ```

   **请确保按照上述要求下载并安装了 miniconda。**

   

   一旦我们安装了 Miniconda，我们首先来安装 Jupyter，这个很好用。

   ```bash
   conda install -y jupyter
   ```

2. 以上我们完成了 Python 的安装，安装 Pytorch的时候我们需要确保Pytorch 和 Python版本对应。最好实现这个要求的方法是使用 Anaconda 的环境。我们创建的每个 Anaconda的环境都可以独立的拥有自己的 Python 版本，驱动以及 Python 的库。我建议大家创建一个独立的环境来进行测试和管理一个单独的大项目。

   使用下面的命令来创建我们自己的环境。我讲这个环境称为"**torch_mac**",我们可以任意的命令自己的环境的名字。我们通过下面的YML 格式文件来进行环境的创建。你可以通过这个链接来获得这个文件，对于 Mac(Apple silicon)的安装命令如下：

   ```bash
   conda env create -f torch-conda.yml
   ```

   为了进入这个环境，我们使用下面的命令：

   ```bash
   conda activate torch
   ```

## 注册我们的环境

再一次请确保你使用了`conda activate`命令来已经激活了你的环境，如果已经激活，应该显示如下：

```bash
Shanglin:~ lynn$ conda activate torch_mac
(torch_mac) Shanglin:~ lynn$ conda --version
conda 23.9.0
(torch_mac) Shanglin:~ lynn$ python --version
Python 3.9.18
```

下面的命令可以注册我们的**pytorch**环境。

```bash
python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (torch)"
```

然后会显示：

```bash
(torch_mac) Shanglin:~ lynn$ python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (torch)"
Installed kernelspec pytorch in /Users/lynn/Library/Jupyter/kernels/pytorch
```

## 验证我们的环境：

```python
(torch_mac) Shanglin:~ lynn$ python
>>> import torch
>>> torch.backends.mps.is_built()
True
>>> 
```

注意，之前可能是使用`getattr(torch, 'has_mps', False)   `这个命令来验证，但是现在torch 官网给出了这个提示，*has_mps' is deprecated, please use 'torch.backends.mps.is_built()*。

所以我们尽量使用 `torch.backends.mps.is_built()`这个命令来验证自己安装的的 torch 是否支持 Mac 独有的MPS。

或者你直接执行这个链接下面的我写好的Jupyter 脚本。

## 尝试训练一个简单的神经网络

请先下载好[mnist数据集](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)到自己的文件夹里：`./data/mnist/mnist.pkl.gz`

然后执行下面的代码：

```python
from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import torch
import time

import torch.nn.functional as F
from torch import nn
from torch import optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

# 更新URL和文件名（如果有必要的话）
URL = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

loss_func = F.cross_entropy

def model(xb):
    return xb.mm(weights) + bias

bs = 500
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True) 
bs = 500
bias = torch.zeros(10, requires_grad=True)



class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out  = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Mnist_NN()

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))

def get_model():
    model = Mnist_NN()
    return model, optim.Adam(model.parameters(), lr=0.001)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()

start_time = time.time()

fit(20, model, loss_func, opt, train_dl, valid_dl)

end_time = time.time()
total_time = end_time - start_time
print(f"Training took {total_time:.2f} seconds")
```

