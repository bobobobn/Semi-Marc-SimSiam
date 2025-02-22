import torch.nn as nn

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.encoder = base_model(out_dim)
        dim_mlp = self.encoder.fc.in_features

        # add mlp projection head
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)

    def forward(self, x):
        return self.encoder(x)
