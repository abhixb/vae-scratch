import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=512, z_dim=128):
        super().__init__()

        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2hid = nn.Linear(h_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2hid2 = nn.Linear(h_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        h = self.relu(self.hid_2hid(h))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        h = self.relu(self.hid_2hid2(h))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x_reconstructed = self.decode(z)
        return mu, sigma, x_reconstructed


if __name__ == "__main__":
    x = torch.randn(4, 3 * 64 * 64)
    vae = VariationalAutoEncoder(input_dim=3 * 64 * 64)
    mu, sigma, x_reconstructed = vae(x)
    print("mu shape:", mu.shape)
    print("sigma shape:", sigma.shape)
    print("reconstructed shape:", x_reconstructed.shape)