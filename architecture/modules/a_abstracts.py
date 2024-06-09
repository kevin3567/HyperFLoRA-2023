import torch
import torch.nn as nn
import torch.nn.functional as F


class InitializableAbstract(nn.Module):
    def init_param(self):
        print("Type {} has not implemented init_param".format(type(self)))
        raise NotImplementedError


class ModelAbstract(nn.Module):  # These don't do anything right now.
    pass


class ComponentAbstract(nn.Module):  # These don't do anything right now.
    pass
