import numpy as np
import scipy.io
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 1
perc_val = 0.2
num_rounds = 1
batch_size = 1024
num_epoch = 2000
num_layers = [2]
num_neurons = [64]
inputs_lib_dynamical = [
    's_norm, U_s'
]

inputs_dim_lib_dynamical = [
    '2 * (inputs_dim - 1)'
]

addr = '..\\..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)
# params_PDE_all = np.zeros((data.num_cells, 3))

metric_mean = dict()
metric_std = dict()
metric_mean['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['test'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['test'] = np.zeros((len(inputs_lib_dynamical), 1))
for l in range(len(inputs_lib_dynamical)):
    inputs_dynamical, inputs_dim_dynamical = inputs_lib_dynamical[l], inputs_dim_lib_dynamical[l]
    layers = num_layers[0] * [num_neurons[0]]
    np.random.seed(1234)
    torch.manual_seed(1234)
    metric_rounds = dict()
    metric_rounds['train'] = np.zeros(num_rounds)
    metric_rounds['val'] = np.zeros(num_rounds)
    metric_rounds['test'] = np.zeros(num_rounds)
    for round in range(num_rounds):
        inputs_dict, targets_dict = func.create_chosen_cells(
            data,
            idx_cells_train=[101, 108, 120],
            idx_cells_test=[116],
            perc_val=perc_val
        )
        inputs_train = inputs_dict['train'].to(device)
        inputs_val = inputs_dict['val'].to(device)
        inputs_test = inputs_dict['test'].to(device)
        targets_train = targets_dict['train'][:, :, 0:1].to(device)
        targets_val = targets_dict['val'][:, :, 0:1].to(device)
        targets_test = targets_dict['test'][:, :, 0:1].to(device)

        inputs_dim = inputs_train.shape[2]
        outputs_dim = 1

        _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
        _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

        train_set = func.TensorDataset(inputs_train, targets_train)  # J_train is a placeholder
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            inputs_dynamical=inputs_dynamical,
            inputs_dim_dynamical=inputs_dim_dynamical
        ).to(device)

        log_sigma_u = torch.randn((), requires_grad=True)
        log_sigma_f = torch.randn((), requires_grad=True)
        log_sigma_f_t = torch.randn((), requires_grad=True)

        criterion = func.My_loss(mode='AdpBal')

        params = ([p for p in model.parameters()] + [log_sigma_u] + [log_sigma_f] + [log_sigma_f_t])
        optimizer = optim.Adam(params, lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.1)
        model, results_epoch = func.train(
            num_epoch=num_epoch,
            batch_size=batch_size,
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_test,
            targets_val=targets_test,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            log_sigma_u=log_sigma_u,
            log_sigma_f=log_sigma_f,
            log_sigma_f_t=log_sigma_f_t
        )

        model.eval()

        U_pred_train, F_pred_train, _ = model(inputs=inputs_train)
        U_pred_train = 1. - U_pred_train
        targets_train = 1. - targets_train
        RMSPE_train = torch.sqrt(torch.mean(((U_pred_train - targets_train) / targets_train) ** 2))

        U_pred_val, F_pred_val, _ = model(inputs=inputs_val)
        U_pred_val = 1. - U_pred_val
        targets_val = 1. - targets_val
        RMSPE_val = torch.sqrt(torch.mean(((U_pred_val - targets_val) / targets_val) ** 2))

        U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
        U_pred_test = 1. - U_pred_test
        targets_test = 1. - targets_test
        RMSPE_test = torch.sqrt(torch.mean(((U_pred_test - targets_test) / targets_test) ** 2))

        metric_rounds['train'][round] = RMSPE_train.detach().cpu().numpy()
        metric_rounds['val'][round] = RMSPE_val.detach().cpu().numpy()
        metric_rounds['test'][round] = RMSPE_test.detach().cpu().numpy()

    metric_mean['train'][l] = np.mean(metric_rounds['train'])
    metric_mean['val'][l] = np.mean(metric_rounds['val'])
    metric_mean['test'][l] = np.mean(metric_rounds['test'])
    metric_std['train'][l] = np.std(metric_rounds['train'])
    metric_std['val'][l] = np.std(metric_rounds['val'])
    metric_std['test'][l] = np.std(metric_rounds['test'])

model.eval()
inputs_test = inputs_dict['test'].to(device)
targets_test = targets_dict['test'][:, :, 0:1].to(device)
U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
U_pred_test = 1. - U_pred_test
targets_test = 1. - targets_test

results = dict()
results['U_true'] = targets_test.detach().cpu().numpy().squeeze()
results['U_pred'] = U_pred_test.detach().cpu().numpy().squeeze()
results['U_t_pred'] = model.U_t.detach().cpu().numpy().squeeze()
results['Cycles'] = inputs_test[:, :, -1:].detach().cpu().numpy().squeeze()
results['Epochs'] = np.arange(0, num_epoch)
results['lambda_U'] = results_epoch['var_U']
results['lambda_F'] = results_epoch['var_F']
torch.save(results, '..\\..\\..\\Results\\4 Presentation\\SoH Estimation\\SoH_CaseB_DeepHPM_AdpBal.pth')
pass
