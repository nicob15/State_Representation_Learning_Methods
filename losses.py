import torch
import torch.distributions as td

def loss_VAE(recon_x, x, mu, log_var, beta=1):
    BCE = bce_loss(recon_x, x)
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