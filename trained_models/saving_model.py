import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path
print(sys.path)


from_address = "{}/bsf_DDQN.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}/validated.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
