import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# for 84x84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}

class DeterministicLinearEncoder(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(DeterministicLinearEncoder, self).__init__()

        self.conv1 = nn.Conv2d(6, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
        self.fc = nn.Linear(32 * out_dim * out_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.encoder(x)

class DeterministicLinearDecoder(nn.Module):
    def __init__(self, z_dim=20):
        super(DeterministicLinearDecoder, self).__init__()

        # decoder part
        out_dim = OUT_DIM[4]
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 6, (3, 3), stride=(2, 2), output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def decoder(self, z):
        z = self.fcz(z)
        z = self.unflatten(z)
        z = self.batch3(z)
        z = self.deconv1(z)
        z = self.deconv2(z)
        z = self.batch4(z)
        z = self.deconv3(z)
        x = self.deconv4(z)
        return x

    def forward(self, x):
        return self.decoder(x)

class DeterministicEncoder(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(DeterministicEncoder, self).__init__()

        self.conv1 = nn.Conv2d(6, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
        self.fc = nn.Linear(32 * out_dim * out_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc(x))
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.encoder(x)

class DeterministicDecoder(nn.Module):
    def __init__(self, z_dim=20):
        super(DeterministicDecoder, self).__init__()

        # decoder part
        out_dim = OUT_DIM[4]
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 6, (3, 3), stride=(2, 2), output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def decoder(self, z):
        z = F.elu(self.fcz(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        x = self.deconv4(z)
        return x

    def forward(self, x):
        return self.decoder(x)

class StochasticEncoder(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(StochasticEncoder, self).__init__()

        self.conv1 = nn.Conv2d(6, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[4]
        self.fc = nn.Linear(32 * out_dim * out_dim, z_dim)
        self.fc1 = nn.Linear(32 * out_dim * out_dim, z_dim)

    def sampling(self, mu, std):
        std = torch.exp(0.5 * torch.log(torch.square(std)))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def encoder(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc(x)
        std = F.relu(self.fc1(x)) + 1e-4
        return mu, std

    def forward(self, x):
        mu, std = self.encoder(x)
        return mu, std, self.sampling(mu, std)

class StochasticDecoder(nn.Module):
    def __init__(self, z_dim=20):
        super(StochasticDecoder, self).__init__()

        # decoder part
        out_dim = OUT_DIM[4]
        self.fcz = nn.Linear(z_dim, 32 * out_dim * out_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, out_dim, out_dim))
        self.deconv1 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 32, (3, 3), stride=(1, 1))
        self.deconv4 = nn.ConvTranspose2d(32, 6, (3, 3), stride=(2, 2), output_padding=(1, 1))
        self.batch4 = nn.BatchNorm2d(32)

    def sampling(self, mu, std):
        std = torch.exp(0.5 * torch.log(torch.square(std)))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        z = F.elu(self.fcz(z))
        z = self.unflatten(z)
        z = self.batch3(z)
        z = F.elu(self.deconv1(z))
        z = F.elu(self.deconv2(z))
        z = self.batch4(z)
        z = F.elu(self.deconv3(z))
        mu = self.deconv4(z)
        std = torch.ones_like(mu).detach()
        o_rec = F.sigmoid(self.sampling(mu, std))
        return mu, std, o_rec

    def forward(self, x):
        return self.decoder(x)


class DeterministicForwardModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicForwardModel, self).__init__()

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        z_next = self.fc2(za)
        return z_next

class StochasticForwardModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1,  max_sigma=1e1, min_sigma=1e-4):
        super(StochasticForwardModel, self).__init__()

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

    def sampling(self, mu, std):
        std = torch.exp(0.5 * torch.log(torch.square(std)))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        mu = self.fc2(za)
        std = F.sigmoid(self.fc3(za))
        std = self.min_sigma + (self.max_sigma - self.min_sigma) * std
        z_next = self.sampling(mu, std)
        return mu, std, z_next

class DeterministicRewardModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicRewardModel, self).__init__()

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        rew = self.fc2(za)
        return rew

class DeterministicRewardModel_no_act(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(DeterministicRewardModel_no_act, self).__init__()

        self.fc = nn.Linear(z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, z):
        z = F.elu(self.fc(z))
        z = F.elu(self.fc1(z))
        rew = self.fc2(z)
        return rew

class DeterministicInverseModel(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, bounded_act='True', scale=2.0):
        super(DeterministicInverseModel, self).__init__()

        self.bounded_act = bounded_act
        self.scale = scale

        self.fc = nn.Linear(z_dim + z_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, a_dim)

    def forward(self, z, z_next):
        zz = torch.cat([z, z_next], dim=1)
        zz = F.elu(self.fc(zz))
        zz = F.elu(self.fc1(zz))
        if self.bounded_act:
            a = self.scale * F.tanh(self.fc2(zz))
        else:
            a = self.fc2(zz)
        return a

class DeterministicActionEncoder(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicActionEncoder, self).__init__()

        self.fc = nn.Linear(z_dim + a_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.elu(self.fc(za))
        za = F.elu(self.fc1(za))
        a_bar = self.fc2(za)
        return a_bar


class DeterministicForwardModelMDPH(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicForwardModelMDPH, self).__init__()

    def forward(self, z, a_bar):
        z_next = z + a_bar
        return z_next