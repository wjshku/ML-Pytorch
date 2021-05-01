## This is about the note I made regarding materials in Lesson2. The whole algorithm is usually divided into 4 parts.
#### 1. Import libraries
import torch
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

### 2. Datasets and dataloaders
We shall divide our data into training dataset, validation dataset and testing dataset using **random.split**

### 3. Training and model
The structure of this part is Build **Customized Class Model**, Write **evalution** function, And last the **Fit** function

### 4. Evaluation and Testing
We shall do the cross validation on validation dataset and output the accuracy model.
One very important element in this part is to have the **History** variable, which stores how the model was improved through training.

### 5. Prediction
The last stage is to use test data to do predictions.

#### At last, I would like to mention some interesting functions and properties mentions in materials.
> 1. .detach(): this was used on loss function in Model to prevent gradients of loss being changed by accident. https://www.kite.com/python/docs/torch.Tensor.detach

> 2. torch.stack(): this was used on validation_epoch_end to combine a list of tensors into one tensor and calculated the mean of accuracy and loss.https://pytorch.org/docs/stable/generated/torch.stack.html

> 3. [model.validation_step(batch) for batch in val_loader]: Nothing special but very useful property of **list()** in **Pytorch**
