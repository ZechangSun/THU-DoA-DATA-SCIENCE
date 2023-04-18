# How to Warp Your Python Code?



**Author**: Zechang Sun

**Email**: szc22@mails.tsinghua.edu.cn



Although it won't directly help with your machine learning model, write a code elegantly will greatly improve your efficient for model training.

An ideal machine learning code should have:

* **clear logic**


* **user-friendly interface**


* **good readability**

### How to Warp Your Python Code?

In this slide, we will give a brief introduction about following packages for more elegant code:

* `pytorch`: deep learning package support for auto-gradient and other common deep learning architectures
* `pytorch_lightning`: deep learning framework with maximal flexibility without sacrificing performance at scale
* `einops`: flexible and powerful tensor operations for readable and reliable code
* `omegaconf`: YAML based hierarchical configuration system
* `importlib`: the implementation of the import statement

### How to Warp Your Python Code? - General Framework for Machine Learning

A machine learning code always consists of at least three parts:

* **Data module**: data loading, preprocessing

* **Model module**: model definition

* **Training module**: optimizer, learning rate scheduler

### How to Warp Your Python Code? - `pytorch`
#### Data Module


```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class YourDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = np.random.randn(2048)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int):
        return self.data[idx]
```


```python
dataset = YourDataset()
train_dataset, val_dataset = random_split(dataset, [1024, 1024])
train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True), \
                                    DataLoader(val_dataset, batch_size=128, shuffle=True)
```

Useful packages for data preprocessing:

* `torchvision`:  consists of popular datasets, model architectures, and common image transformations for computer vision
* `torchaudio`: audio and signal processing with PyTorch
* `torchtext`:  consists of data processing utilities and popular datasets for natural language

## How to Warp Your Python Code? - `pytorch`
### Model Module


```python
from torch import nn

class YourModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.model(x)
```


```python
model = YourModel()
model(torch.rand(1).reshape(-1, 1))
```




    tensor([[-0.4735]], grad_fn=<AddmmBackward0>)



## How to Warp Your Python Code? - `pytorch`

### Training Module


```python
from torch import optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```


```python
num_epochs = 100
for epoch in range(num_epochs):
    for batch in train_dataloader:
        x = batch.reshape(-1, 1).float()
        out = model(x)
        loss = nn.functional.l1_loss(x, out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'epoch: {epoch+1}; loss: {loss.item()}')
    break
```

    epoch: 1; loss: 1.3107006549835205


## How to Warp Your Python Code? - `pytorch_lightning`

### Data Module


```python
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
```

## How to Warp Your Python Code? - `pytorch_lightning`

### Model/Training Module

```python
class LitModel(pl.LightningModule):
    def __init__(self, ...):
        ...
    
    def forward(self, ...):
        ...
    
    def training_step(self, batch, batch_idx):
        ...
    
    def training_step_end(self, ...):
        ...
    
    def training_epoch_end(self, ...):
        ...
    
    def validation_step(self, batch, batch_idx):
        ...
    
    def validation_step_end(self, ...):
        ...
    
    def test_step(self, ...):
        ...
    
    def test_step_end(self, ...):
        ...
    
    def test_epoch_end(self, ...):
        ...
    
    def configure_optimizers(self, ...):
        ...
```


```python
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
```

## How to Warp Your Python Code? - `pytorch_lightning`

### Trainer


```python
# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = DataLoader(dataset)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw/train-images-idx3-ubyte.gz



      0%|          | 0/9912422 [00:00<?, ?it/s]


    Extracting /Users/zechang/Desktop/jupyter/MNIST/raw/train-images-idx3-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw/train-labels-idx1-ubyte.gz



      0%|          | 0/28881 [00:00<?, ?it/s]


    Extracting /Users/zechang/Desktop/jupyter/MNIST/raw/train-labels-idx1-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw/t10k-images-idx3-ubyte.gz



      0%|          | 0/1648877 [00:00<?, ?it/s]


    Extracting /Users/zechang/Desktop/jupyter/MNIST/raw/t10k-images-idx3-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw/t10k-labels-idx1-ubyte.gz



      0%|          | 0/4542 [00:00<?, ?it/s]


    Extracting /Users/zechang/Desktop/jupyter/MNIST/raw/t10k-labels-idx1-ubyte.gz to /Users/zechang/Desktop/jupyter/MNIST/raw




```python
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /Users/zechang/opt/anaconda3/envs/pytorch/lib/python3.8/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:67: UserWarning: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
      warning_cache.warn(
    Missing logger folder: /Users/zechang/Desktop/jupyter/lightning_logs
    
      | Name    | Type       | Params
    ---------------------------------------
    0 | encoder | Sequential | 50.4 K
    1 | decoder | Sequential | 51.2 K
    ---------------------------------------
    101 K     Trainable params
    0         Non-trainable params
    101 K     Total params
    0.407     Total estimated model params size (MB)
    /Users/zechang/opt/anaconda3/envs/pytorch/lib/python3.8/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 10 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(



    Training: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=1` reached.


## How to Warp Your Python Code? - `einops`


```python
from einops import rearrange, reduce, repeat

tensor = torch.rand(1, 2, 3)
rearrange(tensor, 'h w c -> w h c').shape
```




    torch.Size([2, 1, 3])




```python
from einops import rearrange, reduce, repeat

tensor = torch.rand(1, 2, 3)
rearrange(tensor, 'h w (c i) -> w h c i', i=1).shape
```




    torch.Size([2, 1, 3, 1])




```python
reduce(tensor, 'b h w -> b h', 'mean')
```




    tensor([[0.4237, 0.6134]])




```python
repeat(tensor, 'b h w -> b h w c', c=3).shape
```




    torch.Size([1, 2, 3, 3])



## How to Warp Your Python Code? - `omegaconf`


```python
from omegaconf import OmegaConf

conf = OmegaConf.create({"k" : "v", "list" : [1, {"a": "1", "b": "2", 3: "c"}]})
print(OmegaConf.to_yaml(conf))
```

    k: v
    list:
    - 1
    - a: '1'
      b: '2'
      3: c




```python
with open('result.yaml', 'w') as file:
    file.write(OmegaConf.to_yaml(conf))
```


```python
conf = OmegaConf.load('result.yaml')
# Output is identical to the YAML file
print(OmegaConf.to_yaml(conf))
```

    k: v
    list:
    - 1
    - a: '1'
      b: '2'
      3: c




```python
conf = OmegaConf.merge(base_cfg, model_cfg, optimizer_cfg, dataset_cfg)
```

## How to Warp Your Python Code? - `importlib`


```python
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise ValueError("No target function founded!")
    return get_obj_from_str(config['target'])(**config.get("params", dict()))
```


```python

```
