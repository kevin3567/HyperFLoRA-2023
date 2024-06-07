from architecture.modules.a_abstracts import ComponentAbstract
from torch import nn
import torch.nn.functional as F


class LeNetFull(ComponentAbstract):
    def __init__(self, expand_size, out_size):
        super(LeNetFull, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # default is 120
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 100)  # outsize if by default is 100
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, out_size)  # outsize if by default is 10

    def forward(self, x):
        x_interm = self.forward_conv(x)
        x_out = self.forward_out(x_interm)
        return x_out, x_interm

    def forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x_interm = x.view(-1, 16 * 5 * 5)
        return x_interm

    def forward_out(self, x_interm):
        x = self.relu1(self.fc1(x_interm))
        x = self.relu2(self.fc2(x))
        x_out = self.fc3(x)
        return x_out

    def init_param(self):
        # nn.init.uniform_(-(k1**0.5), (k1**0.5))  # check torch init
        # nn.init.uniform_(-(k2**0.5), (k2**0.5))  # check torch init
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


class LeNet(ComponentAbstract):
    def __init__(self, expand_size, out_size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, expand_size)  # default is 120
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(expand_size, out_size)  # outsize if by default is 100

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu1(self.fc1(x))

        x = self.fc2(x)
        return x

    def init_param(self):
        # nn.init.trunc_normal_(self.conv1, mean=0, std=0.3, a=-1, b=1)  # Near the range of sinusoid is cos/sin
        # nn.init.trunc_normal_(self.conv2, mean=0, std=0.3, a=-1, b=1)  # Near the range of sinusoid is cos/sin
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
