import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import functions as func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = torch.load('..\\Settings\\settings_RUL_CaseC.pth')
seq_len = 1
perc_val = 0.2
num_rounds = settings['num_rounds']
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']
num_layers = settings['num_layers']
num_neurons = settings['num_neurons']
inputs_lib_dynamical = settings['inputs_lib_dynamical']
inputs_dim_lib_dynamical = settings['inputs_dim_lib_dynamical']

addr = '..\\..\\SeversonBattery.mat'
data = func.SeversonBattery(addr, seq_len=seq_len)

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
    weights_rounds = [[]] * num_rounds
    for round in range(num_rounds):
        inputs_train_val_ndarray, inputs_test_ndarray, targets_train_val_ndarray, targets_test_ndarray = train_test_split(
            data.inputs_val_ndarray, data.targets_val_ndarray,
            test_size=0.2
        )
        inputs_train_ndarray, inputs_val_ndarray, targets_train_ndarray, targets_val_ndarray = train_test_split(
            inputs_train_val_ndarray, targets_train_val_ndarray,
            test_size=0.25
        )
        inputs_train = torch.from_numpy(inputs_train_ndarray).type(torch.float32).to(device)
        inputs_val = torch.from_numpy(inputs_val_ndarray).type(torch.float32).to(device)
        inputs_test = torch.from_numpy(inputs_test_ndarray).type(torch.float32).to(device)
        targets_train = torch.from_numpy(targets_train_ndarray[:, :, 1:]).type(torch.float32).to(device)
        targets_val = torch.from_numpy(targets_val_ndarray[:, :, 1:]).type(torch.float32).to(device)
        targets_test = torch.from_numpy(targets_test_ndarray[:, :, 1:]).type(torch.float32).to(device)

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
        optimizer = optim.Adam(params, lr=settings['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])
        model, results_epoch = func.train(
            num_epoch=num_epoch,
            batch_size=batch_size,
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_val,
            targets_val=targets_val,
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
        RMSE_train = torch.sqrt(torch.mean(((U_pred_train - targets_train)) ** 2))

        U_pred_val, F_pred_val, _ = model(inputs=inputs_val)
        RMSE_val = torch.sqrt(torch.mean(((U_pred_val - targets_val)) ** 2))

        U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
        RMSE_test = torch.sqrt(torch.mean(((U_pred_test - targets_test)) ** 2))

        metric_rounds['train'][round] = RMSE_train.detach().cpu().numpy()
        metric_rounds['val'][round] = RMSE_val.detach().cpu().numpy()
        metric_rounds['test'][round] = RMSE_test.detach().cpu().numpy()
        weights_rounds[round] = dict()
        weights_rounds[round]['lambda_U'] = results_epoch['var_U']
        weights_rounds[round]['lambda_F'] = results_epoch['var_F']
        weights_rounds[round]['lambda_F_t'] = results_epoch['var_F_t']
        torch.save(weights_rounds, '..\\..\\Results\\3 Adaptive Balancing\\weights_rounds_RUL_CaseC_DeepHPM_AdpBal.pth')

    metric_mean['train'][l] = np.mean(metric_rounds['train'])
    metric_mean['val'][l] = np.mean(metric_rounds['val'])
    metric_mean['test'][l] = np.mean(metric_rounds['test'])
    metric_std['train'][l] = np.std(metric_rounds['train'])
    metric_std['val'][l] = np.std(metric_rounds['val'])
    metric_std['test'][l] = np.std(metric_rounds['test'])
    torch.save(metric_mean, '..\\..\\Results\\3 Adaptive Balancing\\metric_mean_RUL_CaseC_DeepHPM_AdpBal.pth')
    torch.save(metric_std, '..\\..\\Results\\3 Adaptive Balancing\\metric_std_RUL_CaseC_DeepHPM_AdpBal.pth')

pass
