import torch.nn as nn

from models import DeterministicLinearEncoder, DeterministicLinearDecoder, DeterministicEncoder, DeterministicDecoder, \
                   StochasticEncoder, StochasticDecoder, DeterministicForwardModel, StochasticForwardModel, \
                   DeterministicRewardModel, DeterministicRewardModel_no_act, DeterministicInverseModel

class LinearAE(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(LinearAE, self).__init__()

        self.encoder = DeterministicLinearEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = DeterministicLinearDecoder(z_dim=z_dim)

    def forward(self, o):
        z = self.encoder(o)
        o_rec = self.decoder(z)
        return z, o_rec

class AE(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(AE, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = DeterministicDecoder(z_dim=z_dim)

    def forward(self, o):
        z = self.encoder(o)
        o_rec = self.decoder(z)
        return z, o_rec

class VAE(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(VAE, self).__init__()

        self.encoder = StochasticEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = StochasticDecoder(z_dim=z_dim)

    def forward(self, o):
        mu_z, std_z, z = self.encoder(o)
        mu_o, std_o, o_rec = self.decoder(z)
        return z, mu_z, std_z, mu_o, std_o, o_rec

class DeterministicFW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicFW, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.fw_model = DeterministicForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        z_next = self.fw_model(z, a)
        return z, z_next, z_target

class StochasticFW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(StochasticFW, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.fw_model = StochasticForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

    def forward(self, x, a):
        z = self.encoder(x)
        _, _, z_next = self.fw_model(z, a)
        return z, z_next

class DeterministicRW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, use_act='True'):
        super(DeterministicRW, self).__init__()

        self.use_act = use_act

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        if self.use_act:
            self.rw_model = DeterministicRewardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        else:
            self.rw_model = DeterministicRewardModel_no_act(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_next = self.encoder(o_next)
        if self.use_act:
            r = self.rw_model(z, a)
        else:
            r = self.rw_model(z_next)
        return z, r

class DeterministicIN(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, bounded_act='True', scale=2.0):
        super(DeterministicIN, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.in_model = DeterministicInverseModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim, bounded_act=bounded_act,
                                                  scale=scale)

    def forward(self, o, o_next):
        z = self.encoder(o)
        z_next = self.encoder(o_next)
        a = self.in_model(z, z_next)
        return z, a