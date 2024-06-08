import torch
import torch.nn as nn

from architecture.modules.a_abstracts import ComponentAbstract, ModelAbstract


class UserTokenNet(ComponentAbstract):
    def __init__(self, embedding_dim, grad_mode):
        super().__init__()
        self.token = nn.Parameter(torch.randn((1, embedding_dim)))
        self._set_grad_mode(grad_mode)

    def forward(self):
        return self.token

    def init_param(self):
        grad_mode = self.token.requires_grad
        self.token = nn.Parameter(torch.randn_like(self.token))
        self._set_grad_mode(grad_mode)

    def assign_param(self, token_tensor):
        assert self.token.size() == token_tensor.size()
        device = self.token.device
        self.token.data = token_tensor.data.to(device)

    def _set_grad_mode(self, mode):
        self.token.requires_grad = mode


class UserToken_Generator(ModelAbstract):
    def __init__(self, uservec_hypparam):
        super().__init__()
        assert uservec_hypparam["name"] == "uservec"
        self.network = UserTokenNet(
            embedding_dim=uservec_hypparam["embedding_dim"],
            grad_mode=uservec_hypparam["grad_mode"]
        )
        self.init_param()

    def forward(self):
        x_out = self.network()
        return x_out

    def init_param(self):
        self.network.init_param()

    def assign_param(self, token_tensor):
        self.network.assign_param(token_tensor)


if __name__ == "__main__":
    uservec_hypparam = {
        "name": "uservec",
        "embedding_dim": 10,
        "grad_mode": False
    }
    generator = UserToken_Generator(uservec_hypparam=uservec_hypparam)
    token_vec = generator()
    generator.assign_param(torch.ones((1, 10)))
    token_vec_frozen = generator()  # double check if token_vec is actually frozen

    print("Done")
