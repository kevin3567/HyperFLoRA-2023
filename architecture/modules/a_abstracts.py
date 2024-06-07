import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modules Abstracts for Functionalities not Default in nn.Module.
"""


class InitializableAbstract(nn.Module):
    def init_param(self):
        print("Type {} has not implemented init_param".format(type(self)))
        raise NotImplementedError

# highest level, can have functions interfacing with environment
class ModelAbstract(nn.Module):
    pass

# lower level, can only have: __init__, forward, and init_param (and sub-functions supporting these implementation)
class ComponentAbstract(nn.Module):
    pass
