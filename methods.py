import torch.nn as nn

from models import DeterministicLinearEncoder, DeterministicLinearDecoder, DeterministicEncoder, DeterministicDecoder, \
                   StochasticEncoder, StochasticDecoder, DeterministicForwardModel, StochasticForwardModel, \
                   DeterministicRewardModel, DeterministicRewardModel_no_act, DeterministicInverseModel, \
                   DeterministicForwardModelMDPH, DeterministicActionEncoder

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

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        mu_next, std_next, z_next = self.fw_model(z, a)
        return z, z_next, mu_next, std_next, z_target

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

class AE_DeterministicFW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(AE_DeterministicFW, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = DeterministicDecoder(z_dim=z_dim)
        self.fw_model = DeterministicForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        z_next = self.fw_model(z, a)
        o_rec = self.decoder(z)
        return z, o_rec, z_next, z_target

class AE_DeterministicRW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, use_act='True'):
        super(AE_DeterministicRW, self).__init__()

        self.use_act = use_act

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = DeterministicDecoder(z_dim=z_dim)
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
        o_rec = self.decoder(z)
        return z, o_rec, r

class AE_DeterministicIN(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(AE_DeterministicIN, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.decoder = DeterministicDecoder(z_dim=z_dim)
        self.in_model = DeterministicInverseModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

    def forward(self, o, o_next):
        z = self.encoder(o)
        z_next = self.encoder(o_next)
        a = self.in_model(z, z_next)
        o_rec = self.decoder(z)
        return z, o_rec, a

class DeterministicFWRW(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, use_act='True'):
        super(DeterministicFWRW, self).__init__()

        self.use_act = use_act

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.fw_model = DeterministicForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        if self.use_act:
            self.rw_model = DeterministicRewardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        else:
            self.rw_model = DeterministicRewardModel_no_act(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        z_next = self.fw_model(z, a)
        if self.use_act:
            r = self.rw_model(z, a)
        else:
            r = self.rw_model(z_target)
        return z, z_next, z_target, r

class DeterministicFWRWIN(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1, use_act='True'):
        super(DeterministicFWRWIN, self).__init__()

        self.use_act = use_act

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.fw_model = DeterministicForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        if self.use_act:
            self.rw_model = DeterministicRewardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        else:
            self.rw_model = DeterministicRewardModel_no_act(z_dim=z_dim, h_dim=h_dim)
        self.in_model = DeterministicInverseModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)

    def forward(self, o, a, o_next):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        z_next = self.fw_model(z, a)
        if self.use_act:
            r = self.rw_model(z, a)
        else:
            r = self.rw_model(z_target)
        a = self.in_model(z, z_target)
        return z, z_next, z_target, r, a

class EncoderCL(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(EncoderCL, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o, o_neg):
        z = self.encoder(o)
        z_neg = self.encoder(o_neg)
        return z, z_neg

class EncPriors(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(EncPriors, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o1, o1_next, o2, o2_next):
        z1 = self.encoder(o1)
        z1_next = self.encoder(o1_next)
        z1d = z1_next - z1
        z2 = self.encoder(o2)
        z2_next = self.encoder(o2_next)
        z2d = z2_next - z2
        return z1, z1_next, z1d, z2, z2_next, z2d

class DeterministicMDPH(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(DeterministicMDPH, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.action_encoder = DeterministicActionEncoder(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        self.fw_model = DeterministicForwardModelMDPH(z_dim=z_dim, h_dim=h_dim)
        self.rw_model = DeterministicRewardModel_no_act(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o, a, o_next, o_neg):
        z = self.encoder(o)
        z_target = self.encoder(o_next)
        z_neg = self.encoder(o_neg)
        a_bar = self.action_encoder(z, a)
        z_next = self.fw_model(z, a_bar)
        r = self.rw_model(z_target)
        return z, z_next, z_target, z_neg, r, a_bar

class EncDeepBisimulation(nn.Module):
    def __init__(self, z_dim=20, h_dim=256, a_dim=1):
        super(EncDeepBisimulation, self).__init__()

        self.encoder = DeterministicEncoder(z_dim=z_dim, h_dim=h_dim)
        self.fw_model = StochasticForwardModel(z_dim=z_dim, h_dim=h_dim, a_dim=a_dim)
        self.rw_model = DeterministicRewardModel_no_act(z_dim=z_dim, h_dim=h_dim)

    def forward(self, o1, a1, o1_next, o2, a2, o2_next):
        z1 = self.encoder(o1)
        z1_target = self.encoder(o1_next)
        z2 = self.encoder(o2)
        z2_target = self.encoder(o2_next)
        mu1, std1, z1_next = self.fw_model(z1, a1)
        mu2, std2, z2_next = self.fw_model(z2, a2)
        r = self.rw_model(z1_next)
        return z1, z1_target, z1_next, mu1, std1, z2, z2_target, z2_next, mu2, std2, r