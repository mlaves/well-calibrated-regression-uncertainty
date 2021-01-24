import numpy as np
import torch
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from uce import uceloss
from calibration_plots import plot_uncert
from utils import nll_criterion_gaussian
from calibration import Scaler, AuxModel, train_scaler, train_aux
from tqdm import tqdm


def test(out_path, model, base_model, calib_loader, test_loader, outlier_freq=0.005):
    device = next(model.parameters()).device
    mus_calib = []
    vars_calib = []
    logvars_calib = []
    targets_calib = []

    with torch.no_grad():
        for (data, target) in tqdm(calib_loader):
            data, target = data.to(device), target.to(device)

            mu, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True)

            mus_calib.append(mu.detach())
            vars_calib.append(var_bayesian.detach())
            logvars_calib.append(logvar.detach())
            targets_calib.append(target.detach())

    mu_calib = torch.cat(mus_calib, dim=0).clamp(0, 1)
    var_calib = torch.cat(vars_calib, dim=0)
    logvar_calib = torch.cat(logvars_calib, dim=0)
    target_calib = torch.cat(targets_calib, dim=0)

    print("l1 =", torch.nn.functional.l1_loss(mu_calib, target_calib).item())

    err_calib = (target_calib-mu_calib).pow(2).mean(dim=1, keepdim=True).sqrt()

    uncertainty = 'total'

    uncert_calib_aleatoric = logvar_calib.exp().mean(dim=1, keepdim=True)
    uncert_calib_epistemic = var_calib.mean(dim=1, keepdim=True)

    if uncertainty == 'aleatoric':
        uncert_calib = uncert_calib_aleatoric.sqrt().clamp(0, 1)
    elif uncertainty == 'epistemic':
        uncert_calib = uncert_calib_epistemic.sqrt().clamp(0, 1)
    else:
        uncert_calib = (uncert_calib_aleatoric + uncert_calib_epistemic).sqrt().clamp(0, 1)  # total

    print('rmse_calib', (target_calib-mu_calib).pow(2).sum(dim=1, keepdim=True).mean().sqrt().item())
    print('uncert_calib', uncert_calib.mean().item())
    print('uncert_calib_aleatoric', uncert_calib_aleatoric.sqrt().mean().item())
    print('uncert_calib_epistemic', uncert_calib_epistemic.sqrt().mean().item())

    fig, ax = plt.subplots(1)
    ax.plot(uncert_calib.cpu(), err_calib.cpu()[:, 0], '.')

    max_val = max(err_calib.max().item(), uncert_calib.max().item())
    ax.plot([0, max_val], [0, max_val], '--')
    ax.set_xlabel('uncert')
    ax.set_ylabel('err')
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_scatter.pdf")

    # calculate optimal T
    S = err_calib.sum() / uncert_calib.sum()
    print('S', S)

    # find optimal S
    scaler = Scaler(init_S=S.item()).to(device)
    train_scaler(scaler, nll_criterion_gaussian, mu_calib, uncert_calib, target_calib)
    print('scaler.S', scaler.S.item())

    # find optimal aux
    aux = AuxModel(1).to(device)
    train_aux(aux, nll_criterion_gaussian, mu_calib, uncert_calib, target_calib)

    aux.train()
    print('nll_criterion_gaussian')
    print(nll_criterion_gaussian(mu_calib, uncert_calib.pow(2).log(), target_calib).item())
    print(nll_criterion_gaussian(mu_calib, (S*uncert_calib).pow(2).log(), target_calib).item())
    print(nll_criterion_gaussian(mu_calib, scaler(uncert_calib).pow(2).log(), target_calib).item())
    print(nll_criterion_gaussian(mu_calib, aux(uncert_calib), target_calib).item())
    print("")
    aux.eval()

    print("mse_loss")
    print(torch.nn.functional.mse_loss(uncert_calib, err_calib, reduction='sum').item())
    print(torch.nn.functional.mse_loss((S*uncert_calib), err_calib, reduction='sum').item())
    print(torch.nn.functional.mse_loss(scaler(uncert_calib), err_calib, reduction='sum').item())
    print(torch.nn.functional.mse_loss(aux(uncert_calib), err_calib, reduction='sum').item())
    print("")

    uce_uncal, err_in_bin_uncal, avg_sigma_in_bin_uncal, _ = uceloss(err_calib, uncert_calib)
    plot_uncert(err_in_bin_uncal.cpu(), avg_sigma_in_bin_uncal.cpu())
    plt.title(str(uce_uncal.item()))
    print('uce_uncal', uce_uncal.item())
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_calib_uncal.pdf")

    uce_s, err_in_bin_s, avg_sigma_in_bin_s, _ = uceloss(err_calib, (S*uncert_calib))
    plot_uncert(err_in_bin_s.cpu(), avg_sigma_in_bin_s.cpu())
    print('uce_s', uce_s.item())
    plt.title(str(uce_s.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_calib_s.pdf")

    uce_cal, err_in_bin_cal, avg_uncert_in_bin_cal, _ = uceloss(err_calib, scaler(uncert_calib))
    plot_uncert(err_in_bin_cal.cpu(), avg_uncert_in_bin_cal.cpu())
    print('uce_cal', uce_cal.item())
    plt.title(str(uce_cal.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_calib_cal.pdf")

    uce_aux, err_in_bin_aux, avg_uncert_in_bin_aux, _ = uceloss(err_calib, aux(uncert_calib))
    plot_uncert(err_in_bin_aux.cpu(), avg_uncert_in_bin_aux.cpu())
    print('uce_aux', uce_aux.item())
    plt.title(str(uce_aux.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_calib_aux.pdf")

    print('')

    mus_test = []
    vars_test = []
    logvars_test = []
    targets_test = []

    with torch.no_grad():
        for (data, target) in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            mu, logvar, var_bayesian = model(data, dropout=True, mc_dropout=True)

            mus_test.append(mu.detach())
            vars_test.append(var_bayesian.detach())
            logvars_test.append(logvar.detach())
            targets_test.append(target.detach())

    mu_test = torch.cat(mus_test, dim=0).clamp(0, 1)
    var_test = torch.cat(vars_test, dim=0)
    logvar_test = torch.cat(logvars_test, dim=0)
    target_test = torch.cat(targets_test, dim=0)

    print("l1 =", torch.nn.functional.l1_loss(mu_test, target_test).item())

    err_test = (target_test-mu_test).pow(2).mean(dim=1, keepdim=True).sqrt()

    uncert_aleatoric_test = logvar_test.exp()
    uncert_epistemic_test = var_test.mean(dim=1, keepdim=True)

    if uncertainty == 'aleatoric':
        uncert_test = uncert_aleatoric_test.sqrt().clamp(0, 1)
    elif uncertainty == 'epistemic':
        uncert_test = uncert_epistemic_test.sqrt().clamp(0, 1)
    else:
        uncert_test = (uncert_aleatoric_test + uncert_epistemic_test).sqrt().clamp(0, 1)  # total

    print('rmse_test', (target_test-mu_test).pow(2).sum(dim=1, keepdim=True).mean().sqrt().item())
    print('uncert_test', uncert_test.mean().item())
    print('uncert_aleatoric_test', uncert_aleatoric_test.sqrt().mean().item())
    print('uncert_epistemic_test', uncert_epistemic_test.sqrt().mean().item())

    aux.train()
    print('nll_criterion_gaussian')
    print(nll_criterion_gaussian(mu_test, uncert_test.pow(2).log(), target_test).item())
    print(nll_criterion_gaussian(mu_test, (S*uncert_test).pow(2).log(), target_test).item())
    print(nll_criterion_gaussian(mu_test, scaler(uncert_test).pow(2).log(), target_test).item())
    print(nll_criterion_gaussian(mu_test, aux(uncert_test), target_test).item())
    print('')
    aux.eval()

    print('mse_loss')
    print(torch.nn.functional.mse_loss(uncert_test, err_test, reduction='sum').item())
    print(torch.nn.functional.mse_loss((S*uncert_test), err_test, reduction='sum').item())
    print(torch.nn.functional.mse_loss(scaler(uncert_test), err_test, reduction='sum').item())
    print(torch.nn.functional.mse_loss(aux(uncert_test), err_test, reduction='sum').item())
    print('')

    uce_uncal_test, err_in_bin_uncal_test, avg_sigma_in_bin_uncal_test, freq_in_bin_uncal_test = uceloss(err_test, uncert_test)
    plot_uncert(err_in_bin_uncal_test.cpu(), avg_sigma_in_bin_uncal_test.cpu(), freq_in_bin_uncal_test, outlier_freq)
    plt.title(str(uce_uncal_test.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_test_uncal.pdf")
    print('uce_uncal', uce_uncal_test.item())

    uce_s_test, err_in_bin_s_test, avg_sigma_in_bin_s_test, freq_in_bin_s_test = uceloss(err_test, S*uncert_test)
    plot_uncert(err_in_bin_s_test.cpu(), avg_sigma_in_bin_s_test.cpu(), freq_in_bin_s_test, outlier_freq)
    plt.title(str(uce_s_test.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_test_s.pdf")
    print('uce_s', uce_s_test.item())

    uce_cal_test, err_in_bin_cal_test, avg_sigma_in_bin_cal_test, freq_in_bin_cal_test = uceloss(err_test, scaler(uncert_test))
    plot_uncert(err_in_bin_cal_test.cpu(), avg_sigma_in_bin_cal_test.cpu(), freq_in_bin_cal_test, outlier_freq)
    plt.title(str(uce_cal_test.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_test_cal.pdf")
    print('uce_cal', uce_cal_test.item())

    uce_aux_test, err_in_bin_aux_test, avg_sigma_in_bin_aux_test, freq_in_bin_aux_test = uceloss(err_test, aux(uncert_test))
    plot_uncert(err_in_bin_aux_test.cpu(), avg_sigma_in_bin_aux_test.cpu(), freq_in_bin_aux_test, outlier_freq)
    plt.title(str(uce_aux_test.item()))
    plt.tight_layout()
    plt.savefig(f"./{out_path}/{base_model}_test_aux.pdf")
    print('uce_aux', uce_aux_test.item())

    np.savez(f"./{out_path}/results_{base_model}",
             uce_uncal_test=uce_uncal_test.cpu().numpy(),
             err_in_bin_uncal_test=err_in_bin_uncal_test.cpu().numpy(),
             avg_sigma_in_bin_uncal_test=avg_sigma_in_bin_uncal_test.cpu().numpy(),
             freq_in_bin_uncal_test=freq_in_bin_uncal_test.cpu().numpy(),
             uce_s_test=uce_s_test.cpu().numpy(),
             err_in_bin_s_test=err_in_bin_s_test.cpu().numpy(),
             avg_sigma_in_bin_s_test=avg_sigma_in_bin_s_test.cpu().numpy(),
             freq_in_bin_s_test=freq_in_bin_s_test.cpu().numpy(),
             uce_cal_test=uce_cal_test.detach().cpu().numpy(),
             err_in_bin_cal_test=err_in_bin_cal_test.cpu().numpy(),
             avg_sigma_in_bin_cal_test=avg_sigma_in_bin_cal_test.cpu().numpy(),
             freq_in_bin_cal_test=freq_in_bin_cal_test.cpu().numpy(),
             uce_aux_test=uce_aux_test.detach().cpu().numpy(),
             err_in_bin_aux_test=err_in_bin_aux_test.cpu().numpy(),
             avg_sigma_in_bin_aux_test=avg_sigma_in_bin_aux_test.cpu().numpy(),
             freq_in_bin_aux_test=freq_in_bin_aux_test.cpu().numpy(),
             )

    print("#"*80)
    plt.close('all')
    sys.stdout.flush()
