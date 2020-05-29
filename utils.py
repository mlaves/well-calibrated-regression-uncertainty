# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch

__all__ = ['leaky_relu1',
           'normal_init',
           'xavier_normal_init',
           'kaiming_normal_init', 'kaiming_uniform_init',
           'nll_criterion_gaussian', 'nll_criterion_laplacian',
           'save_current_snapshot']


def leaky_relu1(x, slope=0.1, a=1):
    x = torch.nn.functional.leaky_relu(x, negative_slope=slope)
    x = -torch.nn.functional.leaky_relu(-x+a, negative_slope=slope)+a
    return x


def normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight.data)


def xavier_normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)


def kaiming_normal_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)


def kaiming_uniform_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)


def nll_criterion_gaussian(mu, logvar, target, reduction='mean'):
    loss = torch.exp(-logvar) * torch.pow(target-mu, 2).mean(dim=1, keepdim=True) + logvar
    return loss.mean() if reduction == 'mean' else loss.sum()


def nll_criterion_laplacian(mu, logsigma, target, reduction='mean'):
    loss = torch.exp(-logsigma) * torch.abs(target-mu).mean(dim=1, keepdim=True) + logsigma
    return loss.mean() if reduction == 'mean' else loss.sum()


def save_current_snapshot(base_model, likelihood, dataset, e, model, optimizer_net, train_losses, valid_losses):
    filename = f"./snapshots/{base_model}_{likelihood}_{dataset}_{e}.pth.tar"
    print(f"Saving at epoch: {e}")
    torch.save({
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer': optimizer_net.state_dict(),
        'train_losses': train_losses,
        'val_losses': valid_losses,
    }, filename)
