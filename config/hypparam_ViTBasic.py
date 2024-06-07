from architecture.modules.c_vitbasic import ViTBasic_Classifier

vitbasic_hypparam = {  # cifar is 32 * 32
    "name": "vitbasic",
    "img_size": 32,
    "patch_size": 4,
    "in_chans": 3,
    "n_output": 100,
    "embed_dim": 64,
    "depth": 4,
    "n_heads": 4,
    "mlp_ratio": 4.,
    "qkv_bias": True,
    "p": 0.,
    "attn_p": 0.,
}


def create_architectures(num_classes):
    vitbasic_hypparam["n_output"] = num_classes
    return ViTBasic_Classifier(
        vitbasic_hypparam=vitbasic_hypparam,
    )


if __name__ == "__main__":
    import torch
    x = torch.randn((15, 3, 32, 32))  # batch, channel, height, width
    model = create_architectures(num_classes=127)
    y_val = model(x)
    print("Done")
