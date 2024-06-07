import torch
import torch.nn as nn
import torch.nn.utils as utils

from collections import OrderedDict
import re

from architecture.modules.a_abstracts import ModelAbstract, ComponentAbstract

# TODO generalize the adapt to lora_hypnet with any network (long term goal)
# TODO: functionalize the k.replace(",","/") to convert names
# TODO: functionalize the k.replace("/",",") to revert names

# tgweights: TargetModel weights
# hpweights: HyperModel weights

def collect_tgweights(bidx, _inelement_tgweights):
    return OrderedDict({k: v[bidx] for k, v in _inelement_tgweights.items()})


class LeNetLora_HypNet(ComponentAbstract):
    def __init__(
            self,
            tgmodel,
            embedding_dim,
            hidden_dim,
            n_hidden,):
        super().__init__()

        # initialize hypnet parameters (simple MLP)
        _layers = [nn.Linear(embedding_dim, hidden_dim)]  # create the embedding dimension
        for _ in range(n_hidden):
            _layers.append(nn.ReLU())
            _layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*_layers)
        del _layers

        # get adaptor weights here (should also account for bias)
        self.tgweight_size_dict = {}
        self.tgweight_module_dict = {}
        for k, v in tgmodel.state_dict().items():
            # check if key is an adaptor (check if key has the following component)
            if re.search(r'fc[0-9]+_adaptor', k):
                k = k.replace(".", "/")
                self.tgweight_module_dict[k] = torch.nn.Linear(hidden_dim, v.numel())
                self.tgweight_size_dict[k] = v.size()
        self.tgweight_module_dict = nn.ModuleDict(self.tgweight_module_dict)

        self.init_param()

    def forward(self, x):
        # x should be a batch of client embedding (batch_size, embedding_dim)
        features = self.mlp(x)

        tgweights = self.batchify_output(features)

        return tgweights  # this should be a list of dicts

    def batchify_output(self, features):
        batch_size = features.size(0)
        # TODO: Declare these the weight names at the __init__ function.
        # TODO: Link the parameter name to the target_weight generation modules.
        # TODO: Dynamic declaration of target_weight generation modules based on target model architecture.

        _inelement_tgweights = OrderedDict({
            k.replace("/", "."): v(features).view(-1, *self.tgweight_size_dict[k]) for k, v in
            self.tgweight_module_dict.items()
        })
        tgweights = [collect_tgweights(bidx, _inelement_tgweights) for bidx in range(batch_size)]
        return tgweights

    def get_tgweight_keys(self):
        return [k.replace("/", ".") for k in self.tgweight_module_dict.keys()]

    def init_param(self):
        print("Type {} has not implemented init_param".format(type(self)))


class LeNetLora_HypNet_Regressor(ModelAbstract):
    def __init__(self, lenetlora_hypnet_hypparam, tgmodel):
        super().__init__()
        assert lenetlora_hypnet_hypparam["name"] == "lenetlora_hypnet"
        self.network = LeNetLora_HypNet(
            tgmodel=tgmodel,
            embedding_dim=lenetlora_hypnet_hypparam["embedding_dim"],
            hidden_dim=lenetlora_hypnet_hypparam["hidden_dim"],
            n_hidden=lenetlora_hypnet_hypparam["n_hidden"],
        )
        self.init_param()

    def forward(self, x):
        x_out = self.network(x)
        return x_out

    def get_tgweight_keys(self):
        return self.network.get_tgweight_keys()

    def init_param(self):
        self.network.init_param()


if __name__ == "__main__":
    from architecture.modules.c_lenetlora import LeNetLora_Classifier
    from architecture.modules.c_usertoken import UserToken_Generator

    lenetlora_hypparam = {  # cifar 10 is 32 * 32
        "name": "lenetlora",  # fixed param for now
        "rank_size": 7,
        "out_size": 10,  # 10
    }
    tgmodel = LeNetLora_Classifier(lenetlora_hypparam)

    lenetlora_hypnet_hypparam = {  # cifar 10 is 32 * 32
        "name": "lenetlora_hypnet",  # fixed param for now=
        "embedding_dim": 32,
        "hidden_dim": 100,
        "n_hidden": 3,
    }
    hypmodel = LeNetLora_HypNet_Regressor(lenetlora_hypnet_hypparam=lenetlora_hypnet_hypparam, tgmodel=tgmodel)

    uservec_hypparam = {
        "name": "uservec",
        "embedding_dim": 32,
        "grad_mode": False
    }
    num_users = 100
    user_idxs_1 = torch.IntTensor([11, 25, 33])
    user_idxs_2 = torch.IntTensor([22, 21, 35, 7, 25, 11])
    usertoken_generators = [UserToken_Generator(uservec_hypparam=uservec_hypparam) for _ in range(num_users)]

    user_tokens_1 = torch.cat([usertoken_generators[idx]() for idx in user_idxs_1], dim=0)
    user_tokens_2 = torch.cat([usertoken_generators[idx]() for idx in user_idxs_2], dim=0)

    tgmodel_weights_1 = hypmodel(user_tokens_1)
    w_tgmodel_form = tgmodel.state_dict()
    for k, v in tgmodel_weights_1[0].items():
        w_tgmodel_form[k] = v
    tgmodel.load_state_dict(w_tgmodel_form)
    tgmodel_weights_2 = hypmodel(user_tokens_2)
    print("Compare tgmodel_weights_1[1] and tgmodel_weights_2[4] by evaluation")
    # [torch.allclose(a[1], b[1], atol=0.0000001) for a, b in
    #  zip(tgmodel_weights_1[1].items(), tgmodel_weights_2[4].items())]
    tgmodel_assignable_keys = hypmodel.get_tgweight_keys()

    print("Done")
