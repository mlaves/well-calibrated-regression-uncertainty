import torch


class Scaler(torch.nn.Module):
    def __init__(self, init_S=1.0):
        super().__init__()
        self.S = torch.nn.Parameter(torch.tensor([init_S]))

    def forward(self, x):
        return self.S.mul(x)


class AuxModel(torch.nn.Module):
    def __init__(self, channels, hidden=16):
        super().__init__()
        self.linear1 = torch.nn.Linear(channels, hidden, bias=True)
        self.fc = torch.nn.Linear(hidden, channels, bias=True)

    def forward(self, x):
        x = 2 * (x.log())
        y = self.linear1(x).relu()
        y = self.fc(y)

        if self.training:
            return y
        else:
            return (0.5 * y).exp()


def train_scaler(scaler, criterion, mu_calib, uncert_calib, target_calib):
    s_opt = torch.optim.LBFGS([scaler.S], lr=3e-4, max_iter=100)

    def closure():
        s_opt.zero_grad()

        loss = criterion(mu_calib, scaler(uncert_calib).pow(2).log(), target_calib)

        loss.backward()
        return loss

    s_opt.step(closure)


def train_aux(aux, criterion, mu_calib, uncert_calib, target_calib):
    # find optimal aux
    aux_opt = torch.optim.Adam(aux.parameters(), lr=3e-4, weight_decay=0)
    lr_scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(aux_opt, patience=100, factor=0.1)

    aux.train()
    for i in range(1000):
        aux_opt.zero_grad()
        loss = criterion(mu_calib, aux(uncert_calib), target_calib)
        loss.backward()
        aux_opt.step()
        lr_scheduler_net.step(loss.item())

    return loss.item()
