import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path
print(sys.path)

'''
from_address = "{}\\DDQN_rwd3.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\bsf_DDQN.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''
'''

from_address = "{}\\TEST_DDQN_rwd3.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\DDQN_rwd3.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''

from_address = "{}\\Abstracted_state_rwd10.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\validated_abstract_indirect.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''

from_address = "{}\\bsf_DDQN.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\bsf_TEST.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''
