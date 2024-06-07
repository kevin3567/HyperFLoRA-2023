import torch.nn as nn


from architecture.modules.a_abstracts import ComponentAbstract
"""
Universal dummy component for checking module functionality. Dimension-invariant feedforward.
"""


class DummyIdentity(ComponentAbstract):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        out = self.identity(x)
        return out

    def init_param(self):
        pass
