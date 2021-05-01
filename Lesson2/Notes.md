## This is about the note I made regarding materials in Lesson2.
## The whole algorithm is usually divided into 4 parts.
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

2. Datasets and dataloaders
We shall divide our data into training dataset, validation dataset and testing dataset using random.split

3. Training and model
4. Evaluation and Testing
5. Prediction
