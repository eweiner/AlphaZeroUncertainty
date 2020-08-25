import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SimpleConv2d(nn.Module):
    """
    Building block of residual blocks: conv, batch_norm, non-linearity
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1, prelu=True):
        super(SimpleConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.prelu = torch.nn.PReLU() if prelu else lambda x: x

    def forward(self, x):
        out = self.prelu(self.batch_norm(self.conv(x)))
        return out
        

class ResidualBlock(nn.Module):

    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.features = nn.ModuleList([SimpleConv2d(n_channels, n_channels),
        SimpleConv2d(n_channels, n_channels, prelu=False)])

    def forward(self, x):
        hidden = x
        for layer in self.features:
            hidden = layer(hidden)
        out = torch.nn.functional.relu(hidden + x)
        return out


class AlphaZeroNet(nn.Module):

    def __init__(self, in_channels, action_space, board_size, hidden_channels=32, n_residual_blocks=4):
        super(AlphaZeroNet, self).__init__()
        self.main_net = nn.ModuleList([SimpleConv2d(in_channels, hidden_channels, kernel_size=(5,5), padding=(2,2))])
        trunk = [ResidualBlock(hidden_channels) for _ in range(n_residual_blocks)]
        self.main_net.extend(trunk)
        self.feauture_extraction = nn.Sequential(*self.main_net)

        self.value_net_layers = nn.ModuleList([SimpleConv2d(hidden_channels, 1, kernel_size=(1,1), padding=0),
        Flatten(),
        nn.Linear(board_size, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()])
        self.value_net = nn.Sequential(*self.value_net_layers)

        self.policy_net_layers = nn.ModuleList([SimpleConv2d(hidden_channels, 2, kernel_size=(1,1), padding=0),
        Flatten(),
        nn.Linear(board_size * 2, action_space)])
        self.policy_net = nn.Sequential(*self.policy_net_layers)


    def forward(self, x):
        features = self.feauture_extraction(x)
        val = self.value_net(features)
        policy = self.policy_net(features)
        return val, policy


if __name__ == "__main__":
    test = AlphaZeroNet(2, 7, 42)
    # test_input = torch.arange(84, dtype=torch.float32).reshape(1, 2, 6, 7)
    # optimizer = torch.optim.Adam(test.parameters())
    # for _ in range(100):
    #     optimizer.zero_grad()
    #     val, policy = test(test_input)
    #     loss = (1 - val)**2 - (torch.FloatTensor([0.9, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02]) * torch.log(torch.nn.functional.softmax(policy.squeeze(0), dim=-1))).sum()
    #     loss.backward()
    #     optimizer.step()
    # val, policy = test(test_input)
    # print(torch.nn.functional.softmax(policy.squeeze(0)))
    # print(val)