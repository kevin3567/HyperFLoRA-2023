from architecture.modules.c_usertoken import UserToken_Generator
from architecture.modules.c_vitlora import ViTLora_Classifier
from architecture.modules.c_vitlora_hypnet import ViTLora_HypNet_Regressor


uservec_hypparam = {
    "name": "uservec",
    "embedding_dim": None,
    "grad_mode": False
}

vitlora_hypparam = {  # cifar is 32 * 32
    "name": "vitlora",
    "img_size": 32,
    "patch_size": 4,
    "in_chans": 3,
    "n_output": None,
    "embed_dim": 64,
    "depth": 4,
    "n_heads": 4,
    "mlp_ratio": 4.,
    "rank_size": 1,
    "fin_adapt": True,
    "qkv_bias": True,
    "p": 0.,
    "attn_p": 0.,
}

vitlora_hypnet_hypparam = {
    "name": "vitlora_hypnet",
    "embedding_dim": None,
    "hidden_dim": 128,
    "n_hidden": 2
}


def create_architectures(num_classes):
    uservec_hypparam["embedding_dim"] = num_classes
    vitlora_hypparam["n_output"] = num_classes
    vitlora_hypnet_hypparam["embedding_dim"] = num_classes
    usermodel = UserToken_Generator(uservec_hypparam=uservec_hypparam)
    tgmodel = ViTLora_Classifier(vitlora_hypparam=vitlora_hypparam)
    hypmodel = ViTLora_HypNet_Regressor(vitlora_hypnet_hypparam=vitlora_hypnet_hypparam,
                                        tgmodel=tgmodel)

    return usermodel, tgmodel, hypmodel


if __name__ == "__main__":
    import torch

    x = torch.randn((15, 3, 32, 32))  # batch, channel, height, width
    usermodel, tgmodel, hypmodel = create_architectures(num_classes=129)
    token = usermodel()
    w_tgmodel = hypmodel(token)
    y_val = tgmodel(x)
    print("Done")
