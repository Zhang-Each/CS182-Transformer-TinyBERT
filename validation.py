import os
import sys

from transformer import Transformer
import sentencepiece as spm
import torch as th
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import json
import capita
import os
from transformer_utils import set_device
import gc
from utils import validate_to_array, model_out_to_list
device = 'cpu'
list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]

root_folder = ""
dataset_folder = "dataset/"
sys.path.append(root_folder)
sp = spm.SentencePieceProcessor()
sp.Load(root_folder + "dataset/wp_vocab10000.model")

