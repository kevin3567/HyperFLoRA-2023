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


class ModelAbstract(nn.Module):  # These don't do anything right now.
    pass


class ComponentAbstract(nn.Module):  # These don't do anything right now.
    pass
