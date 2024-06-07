"""
Modified from https://github.com/huggingface/pytorch-image-models/
and Youtube guide https://www.youtube.com/watch?v=ovB0ddFtzzA&t=9s
"""
import torch
import torch.nn as nn

import architecture.modules.c_adaptor as adaptor
from architecture.modules.a_abstracts import ComponentAbstract, ModelAbstract


def get_xavier_std(weight, gain=1.):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
    xavier_std = gain * (2 / (fan_in + fan_out)) ** 0.5
    return xavier_std


class PatchEmbed(ComponentAbstract):  # unit component
    # === Parameters ===
    # img_siz : int
    # pathc_size : int
    # in_chans : int
    # embed_dim : int
    # === Attributes ===
    # n_patches : int
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # === Parameters ===
        # x: torch.Tensor()
        # === Returns ===
        # torch.Tensor
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

    def init_param(self):
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.normal_(self.proj.bias, std=get_xavier_std(self.proj.weight))


class AttentionLora(ComponentAbstract):  # unit component
    """
    # === Parameters ===
    # dim : int
    # n_heads: bool
    # qkv_bias : bool
    # attn_p : float
    # proj_p: float
    # === Attributes ===
    # scale : float
    # qkv : nn.Linear
    # attn_drop, proj_drop : nn.Dropout
    """

    def __init__(self, dim, n_heads, rank_size, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        # Unconventional joint weighting of qkv, but should work.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 3 is the qkv (concatencated)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        # Lora Adaptors
        self.fcq_adaptor = adaptor.Adaptor(dim, dim, rank_size, bias=False)
        self.fck_adaptor = adaptor.Adaptor(dim, dim, rank_size, bias=False)
        self.fcv_adaptor = adaptor.Adaptor(dim, dim, rank_size, bias=False)
        self.fco_adaptor = adaptor.Adaptor(dim, dim, rank_size, bias=False)

    def forward(self, x):
        # === Parameters ===
        # x : torch.Tensor
        # ===Returns ===
        # torch.Tensor
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # original lora considered weight q, weight k, and weight v, as three single matrix (ignoring heads)
        q = qkv[0] + \
            self.fcq_adaptor(x).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = qkv[1] + \
            self.fck_adaptor(x).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = qkv[2] + \
            self.fcv_adaptor(x).reshape(n_samples, n_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg) + self.fco_adaptor(x)
        x = self.proj_drop(x)
        return x

    def init_param(self):
        nn.init.xavier_normal_(self.qkv.weight)
        nn.init.normal_(self.qkv.bias, std=get_xavier_std(self.qkv.weight))
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.normal_(self.proj.bias, std=get_xavier_std(self.proj.weight))

        self.fcq_adaptor.init_param()
        self.fck_adaptor.init_param()
        self.fcv_adaptor.init_param()
        self.fco_adaptor.init_param()


class MLP(ComponentAbstract):  # unit component
    # === Parameters ===
    # in_features : int
    # hidden_features : int
    # out_features : int
    # p : float
    # === Attribute ===
    # fc : nn.Linear
    # act : nn.GELU
    # fc2 : nn.Linear
    # drop : nn.Dropout

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        # === Parameters ===
        # x : torch.Tensor
        # === Returns ===
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def init_param(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=get_xavier_std(self.fc1.weight))
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=get_xavier_std(self.fc2.weight))


class LoraBlock(ComponentAbstract):  # composite component
    """
    === Parameters ===
    dim : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : bool
    p, attn_p: float

    === Attributes
    norm1, norm2: LayerNorm
    attn: AttentionLora
    mlp: MLP
    """

    def __init__(self, dim, n_heads, mlp_ratio, rank_size, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AttentionLora(dim,
                                  n_heads=n_heads,
                                  rank_size=rank_size,
                                  qkv_bias=qkv_bias,
                                  attn_p=attn_p,
                                  proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            p=p)

    def forward(self, x):
        """
        :parameter:
        x : torch.Tensor
        :return:
        torch.Tensor
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def init_param(self):
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        self.attn.init_param()
        self.mlp.init_param()


class ViTLora(ComponentAbstract):
    """
    :parameters:
    img_size : int
    patch_size : int
    in_chans : int
    n_classes : int
    embed_dim : int
    depth : int
    n_heads : int
    mlp_ratio : float
    qkv_bias : bool
    p, attn_p : float
    :attributes:
    patch_embed : PatchEmbed
    cls_token : nn.Parameter
    pos_emb : nn.Parameter
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
    norm : nn.LayerNorm
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans,
                 n_output,
                 embed_dim,
                 depth,
                 n_heads,
                 mlp_ratio,
                 rank_size,
                 fin_adapt=False,
                 qkv_bias=True,
                 p=0.,
                 attn_p=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_output = n_output
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.rank_size = rank_size
        self.fin_adapt = fin_adapt
        self.qkv_bias = qkv_bias
        self.p = p
        self.attn_p = attn_p
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1,
                        1 + self.patch_embed.n_patches,
                        embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                LoraBlock(dim=embed_dim,
                          n_heads=n_heads,
                          mlp_ratio=mlp_ratio,
                          rank_size=rank_size,
                          qkv_bias=qkv_bias,
                          p=p,
                          attn_p=attn_p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_output)
        if self.fin_adapt:
            self.fcf_adaptor = adaptor.Adaptor(embed_dim, n_output, rank_size, bias=False)

    def forward(self, x):
        """
        :parameter:
        x : torch.Tensor
        :return:
        logits
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        if self.fin_adapt:
            x += self.fcf_adaptor(cls_token_final)
        return x

    def init_param(self):
        gain = 1
        nn.init.zeros_(self.cls_token)
        nn.init.xavier_normal_(self.pos_embed)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.normal_(self.head.bias, std=get_xavier_std(self.head.weight))
        if self.fin_adapt:
            self.fcf_adaptor.init_param()

        self.patch_embed.init_param()
        for block in self.blocks:
            block.init_param()


class ViTLora_Classifier(ModelAbstract):
    def __init__(self, vitlora_hypparam):
        super().__init__()
        assert vitlora_hypparam["name"] == "vitlora"
        self.network = ViTLora(
            img_size=vitlora_hypparam["img_size"],
            patch_size=vitlora_hypparam["patch_size"],
            in_chans=vitlora_hypparam["in_chans"],
            n_output=vitlora_hypparam["n_output"],
            embed_dim=vitlora_hypparam["embed_dim"],
            depth=vitlora_hypparam["depth"],
            n_heads=vitlora_hypparam["n_heads"],
            mlp_ratio=vitlora_hypparam["mlp_ratio"],
            rank_size=vitlora_hypparam["rank_size"],
            fin_adapt=vitlora_hypparam["fin_adapt"],
            qkv_bias=vitlora_hypparam["qkv_bias"],
            p=vitlora_hypparam["p"],
            attn_p=vitlora_hypparam["attn_p"],
        )
        self.init_param()

    def forward(self, x):
        x_out = self.network(x)
        return x_out

    def init_param(self):
        self.network.init_param()


if __name__ == "__main__":
    # create data
    x = torch.randn((15, 3, 32, 32))

    vitlora_hypparam = {  # cifar 10 is 32 * 32
        "name": "vitlora",
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "n_output": 100,
        "embed_dim": 32,
        "depth": 4,
        "n_heads": 4,
        "mlp_ratio": 4.,
        "rank_size": 3,
        "fin_adapt": True,
        "qkv_bias": True,
        "p": 0.,
        "attn_p": 0.,
    }
    classifyer = ViTLora_Classifier(vitlora_hypparam=vitlora_hypparam)
    y_pred = classifyer(x)
    print(y_pred.shape)
    print("Done")
