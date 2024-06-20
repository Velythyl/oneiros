from torch import jit
import torch

net = jit.load('model_scripted.pt')

x=0