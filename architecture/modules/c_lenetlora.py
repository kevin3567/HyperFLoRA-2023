import torch
import torch.nn as nn
import torch.nn.functional as F

import architecture.modules.c_adaptor as adaptor
from architecture.modules.a_abstracts import ComponentAbstract, ModelAbstract


class LeNetLora(ComponentAbstract):
    def __init__(self, rank_size, out_size):
        super(LeNetLora, self).__init__()
        # Standard weights
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # default is 120
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 100)  # outsize if by default is 100
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, out_size)

        # Lora weights
        self.fc1_adaptor = adaptor.Adaptor(16 * 5 * 5, 120, rank_size, bias=False)
        self.fc2_adaptor = adaptor.Adaptor(120, 100, rank_size, bias=False)
        self.fc3_adaptor = adaptor.Adaptor(100, out_size, rank_size, bias=False)

        self.init_param()

    def forward(self, x):
        x_interm = self.forward_conv(x)
        x_out = self.forward_out(x_interm)
        return x_out

    def forward_conv(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x_interm = x.view(-1, 16 * 5 * 5)
        return x_interm

    def forward_out(self, x):
        x = self.relu1(self.fc1(x) + \
                       self.fc1_adaptor(x))
        x = self.relu2(self.fc2(x) + \
                       self.fc2_adaptor(x))
        x = self.fc3(x) + \
            self.fc3_adaptor(x)
        return x

    def init_param(self):
        # Initialize conv weights
        # nn.init.uniform_(-(k1**0.5), (k1**0.5))  # check torch init
        # nn.init.uniform_(-(k2**0.5), (k2**0.5))  # check torch init
        # If we use default conv init, it is faster to redeclare the entire weight.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Initialize fcl weights
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        # Initialize adaptor weights
        self.fc1_adaptor.init_param()
        self.fc2_adaptor.init_param()
        self.fc3_adaptor.init_param()


class LeNetLora_Classifier(ModelAbstract):
    def __init__(self, lenetlora_hypparam):
        super().__init__()
        assert lenetlora_hypparam["name"] == "lenetlora"
        self.network = LeNetLora(
            rank_size=lenetlora_hypparam["rank_size"],
            out_size=lenetlora_hypparam["out_size"]
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

    classifyer = LeNetLora(rank_size=7, out_size=10)
    y_pred = classifyer(x)
    print(y_pred)

    # define parameter
    lenetlora_hypparam = {  # cifar 10 is 32 * 32
        "name": "lenetlora",  # fixed param for now
        "rank_size": 8,
        "out_size": 10,  # 10
    }
    classifyer = LeNetLora_Classifier(lenetlora_hypparam=lenetlora_hypparam)
    y_pred = classifyer(x)
    print(y_pred)