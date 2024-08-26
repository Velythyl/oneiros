import numpy as np
import torch

action_history = torch.load("./action_history.pt").cpu().numpy()
action_history = np.cos(action_history)

action_history = action_history[0,0, -10:]

print(action_history.max())
print(action_history.min())
print(action_history.mean())
i=0