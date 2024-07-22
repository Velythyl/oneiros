from torch import jit
import torch

net = jit.load('model_scripted.pt')

input = torch.randn((1,21 + 7)).cuda()

net(input)
