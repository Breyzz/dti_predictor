# Installing and Importing dependencies (use Google Colab to utilize T4 GPU)

import subprocess, warnings
warnings.filterwarnings('ignore')

def run(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode != 0: print(r.stderr[-1000:])

print("Installing packages... (~3 min)")
run("pip install -q rdkit biopython tqdm matplotlib seaborn scikit-learn pandas")
run("pip install -q torch-geometric")
run("pip install -q fair-esm")

import torch, json, urllib.request, gc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import esm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = Path('data');  DATA.mkdir(exist_ok=True)
Path('checkpoints').mkdir(exist_ok=True)
Path('artifacts').mkdir(exist_ok=True) 

print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
