import torch
import torch.distributions as td

def VAE_loss(rec_o, o, mu, log_var, beta=1.0):
    BCE = torch.nn.functional.binary_cross_entropy(rec_o, o, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE, beta*KLD

def kl_divergence(mu_1, var_1, mu_2, var_2, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    div = td.kl_divergence(p, q)
    div = torch.max(div, div.new_full(div.size(), 3))
    return torch.mean(div)

def kl_divergence_balance(mu_1, var_1, mu_2, var_2, alpha=0.8, dim=1):
    p = td.Independent(td.Normal(mu_1, torch.sqrt(var_1)), dim)
    p_stop_grad = td.Independent(td.Normal(mu_1.detach(), torch.sqrt(var_1.detach())), dim)
    q = td.Independent(td.Normal(mu_2, torch.sqrt(var_2)), dim)
    q_stop_grad = td.Independent(td.Normal(mu_2.detach(), torch.sqrt(var_2.detach())), dim)
    div = alpha * td.kl_divergence(p_stop_grad, q) + (1 - alpha) * td.kl_divergence(p, q_stop_grad)
    div = torch.max(div, div.new_full(div.size(), 3))
    return torch.mean(div)

def loss_loglikelihood(mu, target, var, dim):
    normal_dist = torch.distributions.Independent(torch.distributions.Normal(mu, var), dim)
    return torch.mean(normal_dist.log_prob(target))

def loglikelihood_analitical_loss(mu, target, std):
    loss = 0.5 * ((mu - target) / std).pow(2) + torch.log(std)
    return torch.mean(loss)

def contrastive_loss(z_next, z_neg, hinge=0.5):
    dist = torch.nn.functional.mse_loss(z_next, z_neg)
    zeros = torch.zeros_like(dist)
    neg_loss = torch.max(zeros, hinge - dist)
    return torch.mean(neg_loss)

def mse_loss(x, x_target):
    dist = torch.nn.functional.mse_loss(x, x_target)
    return torch.mean(dist)

def bce_loss(x, x_target):
    bce = torch.nn.functional.binary_cross_entropy(x, x_target, reduction='sum')
    return bce

def temp_coherence(sd, a, alpha=2.0):
    loss = torch.exp(- alpha * torch.norm(sd, p=2, dim=1) * torch.norm(a, p=2, dim=1))
    return torch.mean(loss)

def causality(s1, s2, a1, a2, beta=10.0):
    loss = torch.exp(-torch.norm(s1 - s2, p=2, dim=1)**2) * torch.exp(- beta * torch.norm(a1 - a2, p=2, dim=1)**2)
    return torch.mean(loss)

def proportionality(sd1, sd2, a1, a2, beta=10.0):
    loss = (torch.norm(sd2, p=2, dim=1) - torch.norm(sd1, p=2, dim=1))**2 * torch.exp(-beta * torch.norm(a1 - a2, p=2, dim=1)**2)
    return torch.mean(loss)

def repeatability(s1, s2, sd1, sd2, a1, a2, beta=10.0):
    loss = torch.norm(sd2 - sd1, p=2, dim=1)**2 * torch.exp(-torch.norm(s1 - s2, p=2, dim=1)**2) * torch.exp(-beta * torch.norm(a1 - a2, p=2, dim=1)**2)
    return torch.mean(loss)

def bisimulation_loss(z1, z2, r1, r2, mu1, std1, mu2, std2):
    loss = torch.square(torch.norm(z1 - z2, p=1, dim=1) - torch.norm(r1 - r2, p=1, dim=1) - Wasserstein2(mu1, std1, mu2, std2))
    return torch.mean(loss)

def Wasserstein2(mu1, std1, mu2, std2, gamma=0.5):
    loss = gamma * torch.norm(mu1 - mu2, p=2, dim=1)**2 + torch.norm(std1 - std2, p=2, dim=1)**2
    return loss