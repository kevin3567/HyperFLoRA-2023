import torch
import torch.nn as nn

from architecture.modules.a_abstracts import ComponentAbstract


# TODO: Parallel-Additive Adaptor can also be converted into an Serial Insertion Adaptor, with
#  the latter having parallel Identity and LoraWeights.


class Adaptor(ComponentAbstract):
    def __init__(self, in_dim, out_dim, rank_dim, bias):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank_dim = rank_dim
        self.bias = bias
        self.W_a = nn.Linear(in_dim, rank_dim, bias=bias)
        self.W_b = nn.Linear(rank_dim, out_dim, bias=bias)
        self.init_param()

    def forward(self, x):
        x_compr = self.W_a(x)
        x_expd = self.W_b(x_compr)
        return x_expd

    def init_param(self):
        gain = 1  # hardcode the gain
        nn.init.xavier_normal_(self.W_a.weight)
        nn.init.zeros_(self.W_b.weight)
        if self.bias:  # if bias exists
            # init W_a.bias
            fan_in_a, fan_out_a = nn.init._calculate_fan_in_and_fan_out(self.W_a.weight)
            std_a = gain * (2 / (fan_in_a + fan_out_a)) ** 0.5
            nn.init.normal_(self.W_a.bias, std=std_a)
            # init W_b.bias
            nn.init.zeros_(self.W_b.bias)


if __name__ == "__main__":
    # create data
    z = torch.randn((15, 64))

    adaptor_p = Adaptor(64, 128, 5, bias=True)
    z_tilde = adaptor_p(z)
    print(z_tilde)
