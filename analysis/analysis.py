import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import sys
import time
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import Subset


batch_sizes = [4, 8, 16, 64] # different batch_sizes
learning_rates = [0.1, 0.01, 0.001, 0.0001] # different learning rates
num_epochs = [10, 25, 50, 100]

