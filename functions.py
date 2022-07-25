import numpy as np
import scipy.io
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
# from torch.autograd.gradcheck import zero_gradients
import math


class SeversonBattery:
    def __init__(self, data_addr, seq_len):
        self.data = scipy.io.loadmat(data_addr)
        self.seq_len = seq_len
        self.steps_slices = 1

        self.features = self.data['Features_mov_Flt']
        self.RUL = self.data['RUL_Flt']
        self.PCL = self.data['PCL_Flt']
        self.cycles = self.data['Cycles_Flt']

        self.idx_train_units = self.data['train_ind'].flatten() - 1
        self.idx_val_units = self.data['test_ind'].flatten() - 1
        self.idx_test_units = self.data['secondary_test_ind'].flatten() - 1

        self.inputs = np.hstack((self.features, self.cycles.flatten()[:, None]))
        self.targets = np.hstack((self.PCL, self.RUL))
        self.inputs_dim = self.inputs.shape[1]
        self.targets_dim = self.targets.shape[1]

        self.num_cycles_all = self.data['Num_Cycles_Flt'].flatten()
        self.num_cells = len(self.num_cycles_all)

        self.inputs_units, self.targets_units = create_units(
            data=self.features,
            t=self.cycles,
            RUL=self.targets,
            num_units=self.num_cells,
            len_units=np.squeeze(self.num_cycles_all[:, None])
        )

        self.num_train_units = len(self.idx_train_units)
        self.num_val_units = len(self.idx_val_units)
        self.num_test_units = len(self.idx_test_units)

        self.inputs_train_units = []
        self.targets_train_units = []
        self.inputs_val_units = []
        self.targets_val_units = []
        self.inputs_test_units = []
        self.targets_test_units = []

        for i in range(self.num_train_units):
            self.inputs_train_units.append(self.inputs_units[self.idx_train_units[i]])
            self.targets_train_units.append(self.targets_units[self.idx_train_units[i]])

        for i in range(self.num_val_units):
            self.inputs_val_units.append(self.inputs_units[self.idx_val_units[i]])
            self.targets_val_units.append(self.targets_units[self.idx_val_units[i]])

        for i in range(self.num_test_units):
            self.inputs_test_units.append(self.inputs_units[self.idx_test_units[i]])
            self.targets_test_units.append(self.targets_units[self.idx_test_units[i]])

        self.inputs_train_slices, self.targets_train_slices, self.num_slices_lib_train = create_slices(
            data_units=self.inputs_train_units,
            RUL_units=self.targets_train_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        self.inputs_val_slices, self.targets_val_slices, self.num_slices_lib_val = create_slices(
            data_units=self.inputs_val_units,
            RUL_units=self.targets_val_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        self.inputs_test_slices, self.targets_test_slices, self.num_slices_lib_test = create_slices(
            data_units=self.inputs_test_units,
            RUL_units=self.targets_test_units,
            seq_len_slices=self.seq_len,
            steps_slices=self.steps_slices
        )

        # 生成数组形式
        self.num_slices_train = np.sum(self.num_slices_lib_train)
        self.inputs_train_ndarray = np.zeros((self.num_slices_train, self.seq_len, self.inputs_dim))
        self.targets_train_ndarray = np.zeros((self.num_slices_train, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_train_slices)):
            idx_end = idx_start + self.num_slices_lib_train[i]
            self.inputs_train_ndarray[idx_start:idx_end, :, :] = self.inputs_train_slices[i]
            self.targets_train_ndarray[idx_start:idx_end, :, :] = self.targets_train_slices[i]
            idx_start += self.num_slices_lib_train[i]

        # 生成数组形式
        self.num_slices_val = np.sum(self.num_slices_lib_val)
        self.inputs_val_ndarray = np.zeros((self.num_slices_val, self.seq_len, self.inputs_dim))
        self.targets_val_ndarray = np.zeros((self.num_slices_val, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_val_slices)):
            idx_end = idx_start + self.num_slices_lib_val[i]
            self.inputs_val_ndarray[idx_start:idx_end, :, :] = self.inputs_val_slices[i]
            self.targets_val_ndarray[idx_start:idx_end, :, :] = self.targets_val_slices[i]
            idx_start += self.num_slices_lib_val[i]

        # 生成数组形式
        self.num_slices_test = np.sum(self.num_slices_lib_test)
        self.inputs_test_ndarray = np.zeros((self.num_slices_test, self.seq_len, self.inputs_dim))
        self.targets_test_ndarray = np.zeros((self.num_slices_test, self.seq_len, self.targets_dim))
        idx_start = 0
        for i in range(len(self.inputs_test_slices)):
            idx_end = idx_start + self.num_slices_lib_test[i]
            self.inputs_test_ndarray[idx_start:idx_end, :, :] = self.inputs_test_slices[i]
            self.targets_test_ndarray[idx_start:idx_end, :, :] = self.targets_test_slices[i]
            idx_start += self.num_slices_lib_test[i]

        self.inputs_train_tensor = torch.from_numpy(self.inputs_train_ndarray).type(torch.float32)
        self.inputs_val_tensor = torch.from_numpy(self.inputs_val_ndarray).type(torch.float32)
        self.inputs_test_tensor = torch.from_numpy(self.inputs_test_ndarray).type(torch.float32)
        self.targets_train_tensor = torch.from_numpy(self.targets_train_ndarray).type(torch.float32)
        self.targets_val_tensor = torch.from_numpy(self.targets_val_ndarray).type(torch.float32)
        self.targets_test_tensor = torch.from_numpy(self.targets_test_ndarray).type(torch.float32)


def create_units(data, t, RUL, num_units, len_units):
    data_all = np.hstack((data, t.flatten()[:, None]))
    RUL_all = RUL

    data_list = []
    RUL_list = []

    idx_start = 0
    for i in range(num_units):
        idx_end = idx_start + len_units[i]
        data_list.append(data_all[idx_start:idx_end, :])
        RUL_list.append(RUL_all[idx_start:idx_end, :])
        idx_start += len_units[i]

    return data_list, RUL_list


def create_slices(data_units, RUL_units, seq_len_slices, steps_slices):
    data_slices = []
    RUL_slices = []
    num_slices = np.zeros(len(data_units), dtype=np.int)
    for i in range(len(data_units)):
        num_slices_tmp = int((data_units[i].shape[0] - max(seq_len_slices, steps_slices))
                             / steps_slices) + 1  # 每个unit的slice数量
        data_slices_tmp = np.zeros(
            (num_slices_tmp, seq_len_slices, data_units[0].shape[1]))  # 每个unit的数据划分出的slice
        RUL_slices_tmp = np.zeros((num_slices_tmp, seq_len_slices, RUL_units[0].shape[1]))  # 每个unit的RUL划分出的slice
        idx_start = 0
        for j in range(num_slices_tmp):
            idx_end = idx_start + seq_len_slices
            data_slices_tmp[j, :, :] = data_units[i][idx_start:idx_end, :]
            RUL_slices_tmp[j, :, :] = RUL_units[i][idx_start:idx_end, :]
            idx_start += steps_slices
        data_slices.append(data_slices_tmp)
        RUL_slices.append(RUL_slices_tmp)
        num_slices[i] = num_slices_tmp
    return data_slices, RUL_slices, num_slices


def create_chosen_cells(data, idx_cells_train, idx_cells_test, perc_val):
    inputs_train_slices = []
    inputs_val_slices = []
    inputs_test_slices = []
    targets_train_slices = []
    targets_val_slices = []
    targets_test_slices = []

    for idx in idx_cells_train:
        idx_true = idx - 1
        if idx_true in data.idx_train_units:
            idx_tmp = np.where(data.idx_train_units == idx_true)[0][0]
            inputs_tmp = data.inputs_train_slices[idx_tmp]
            targets_tmp = data.targets_train_slices[idx_tmp]
        if idx_true in data.idx_val_units:
            idx_tmp = np.where(data.idx_val_units == idx_true)[0][0]
            inputs_tmp = data.inputs_val_slices[idx_tmp]
            targets_tmp = data.targets_val_slices[idx_tmp]
        if idx_true in data.idx_test_units:
            idx_tmp = np.where(data.idx_test_units == idx_true)[0][0]
            inputs_tmp = data.inputs_test_slices[idx_tmp]
            targets_tmp = data.targets_test_slices[idx_tmp]
        inputs_tmp_train, inputs_tmp_val, targets_tmp_train, targets_tmp_val = train_test_split(
            inputs_tmp, targets_tmp,
            test_size=perc_val
        )
        inputs_train_slices.append(inputs_tmp_train)
        inputs_val_slices.append(inputs_tmp_val)
        targets_train_slices.append(targets_tmp_train)
        targets_val_slices.append(targets_tmp_val)

    for idx in idx_cells_test:
        idx_true = idx - 1
        if idx_true in data.idx_train_units:
            idx_tmp = np.where(data.idx_train_units == idx_true)[0][0]
            inputs_tmp = data.inputs_train_slices[idx_tmp]
            targets_tmp = data.targets_train_slices[idx_tmp]
        if idx_true in data.idx_val_units:
            idx_tmp = np.where(data.idx_val_units == idx_true)[0][0]
            inputs_tmp = data.inputs_val_slices[idx_tmp]
            targets_tmp = data.targets_val_slices[idx_tmp]
        if idx_true in data.idx_test_units:
            idx_tmp = np.where(data.idx_test_units == idx_true)[0][0]
            inputs_tmp = data.inputs_test_slices[idx_tmp]
            targets_tmp = data.targets_test_slices[idx_tmp]
        inputs_test_slices.append(inputs_tmp)
        targets_test_slices.append(targets_tmp)

    inputs_train_ndarray = np.concatenate((inputs_train_slices), axis=0)
    inputs_val_ndarray = np.concatenate((inputs_val_slices), axis=0)
    inputs_test_ndarray = np.concatenate((inputs_test_slices), axis=0)
    targets_train_ndarray = np.concatenate((targets_train_slices), axis=0)
    targets_val_ndarray = np.concatenate((targets_val_slices), axis=0)
    targets_test_ndarray = np.concatenate((targets_test_slices), axis=0)

    inputs_train_tensor = torch.from_numpy(inputs_train_ndarray).type(torch.float32)
    inputs_val_tensor = torch.from_numpy(inputs_val_ndarray).type(torch.float32)
    inputs_test_tensor = torch.from_numpy(inputs_test_ndarray).type(torch.float32)
    targets_train_tensor = torch.from_numpy(targets_train_ndarray).type(torch.float32)
    targets_val_tensor = torch.from_numpy(targets_val_ndarray).type(torch.float32)
    targets_test_tensor = torch.from_numpy(targets_test_ndarray).type(torch.float32)

    inputs = dict()
    targets = dict()

    inputs['train'] = inputs_train_tensor
    inputs['val'] = inputs_val_tensor
    inputs['test'] = inputs_test_tensor

    targets['train'] = targets_train_tensor
    targets['val'] = targets_val_tensor
    targets['test'] = targets_test_tensor

    return inputs, targets


def standardize_tensor(data, mode, mean=0, std=1):
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 转为2D
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std


def inverse_standardize_tensor(data_norm, mean, std):
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))  # 转为2D
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view((-1, data_norm.shape[-2], data_norm.shape[-1]))
    return data


def fwd_gradients(outputs, inputs):
    grad_outputs = torch.ones_like(outputs).requires_grad_(True)
    g = torch.autograd.grad(
        outputs, inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_outputs_g = torch.ones_like(g)
    doutputs_dinputs = torch.autograd.grad(
        g, grad_outputs,
        grad_outputs=grad_outputs_g,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return doutputs_dinputs


def Verhulst(y, r, K, C):
    return r * (y - C) * (1 - (y - C) / (K - C))



class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


class Neural_Net(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Tanh'):
        super(Neural_Net, self).__init__()

        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim

        self.layers = []
        self.layers.append(nn.Linear(in_features=inputs_dim, out_features=layers[0]))
        nn.init.xavier_normal_(self.layers[-1].weight)
        # self.layers.append(nn.BatchNorm1d(num_features=layers[0]))
        if activation == 'Tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'Sin':
            self.layers.append(Sin())
        self.layers.append(nn.Dropout(p=0.5))
        for l in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[l], out_features=layers[l + 1]))
            nn.init.xavier_normal_(self.layers[-1].weight)
            # self.layers.append(nn.BatchNorm1d(num_features=layers[l+1]))
            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(Sin())
            self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.Linear(in_features=layers[l + 1], out_features=outputs_dim))
        nn.init.xavier_normal_(self.layers[-1].weight)
        self.layers.append(nn.Dropout(p=0.5))
        # self.layers.append(nn.Sigmoid())
        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        self.x = x
        self.x.requires_grad_(True)
        self.x_2D = self.x.contiguous().view((-1, self.inputs_dim))
        NN_out_2D = self.NN(self.x_2D)
        self.u_pred = NN_out_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))

        return self.u_pred


class CapacityNN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        super(CapacityNN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets
        # self.dt, self.params_IRK = dt, params_IRK

        self.log_growth_rate = torch.randn((), requires_grad=True)
        self.log_carrying_capacity = torch.randn((), requires_grad=True)
        self.lb_carrying_capacity = 0.2
        self.ub_carrying_capacity = 1
        self.log_initial_loss = torch.randn((), requires_grad=True)
        self.lb_initial_loss = 0
        self.ub_initial_loss = 1e-1

        self.params_PDE = [self.log_growth_rate] + [self.log_carrying_capacity] + [self.log_initial_loss]

        self.neural_net = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

        self.dynamicNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=1,
            outputs_dim=1,
            layers=layers
        )

    def Net_U0(self, x):
        s = x[:, :, 0: self.inputs_dim - 1]
        t = x[:, :, self.inputs_dim - 1:]

        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])

        U = self.neural_net(torch.cat((s_norm, t_norm), dim=2))
        dU_dt = fwd_gradients(U, t)
        U_2D = U.contiguous().view((-1, self.outputs_dim))
        F = Verhulst(y=U, r=self.growth_rate, K=self.carrying_capacity)
        F_2D = F.contiguous().view((-1, self.outputs_dim))
        U0_2D = U_2D - self.dt * torch.matmul(F_2D, self.params_IRK['alpha'].T)
        U0 = U0_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))
        dU0_dt = fwd_gradients(U0, t)
        return U0, dU0_dt

    def Net_U1(self, x):
        s = x[:, :, 0: self.inputs_dim - 1]
        t = x[:, :, self.inputs_dim - 1:]

        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])

        U = self.neural_net(torch.cat((s_norm, t_norm), dim=2))
        dU_dt = fwd_gradients(U, t)
        U_2D = U.contiguous().view((-1, self.outputs_dim))
        F = Verhulst(y=U, r=self.growth_rate, K=self.carrying_capacity)
        F_2D = F.contiguous().view((-1, self.outputs_dim))
        U1_2D = U_2D + self.dt * torch.matmul(F_2D, (self.params_IRK['beta'] - self.params_IRK['alpha']).T)
        U1 = U1_2D.contiguous().view((-1, self.seq_len, self.outputs_dim))
        dU1_dt = fwd_gradients(U1, t)
        return U1, dU1_dt

    def forward(self, inputs):
        self.growth_rate = torch.exp(-self.log_growth_rate)
        self.carrying_capacity = self.lb_carrying_capacity + (self.ub_carrying_capacity - self.lb_carrying_capacity) * \
                                 torch.sigmoid(self.log_carrying_capacity)
        self.initial_loss = self.lb_initial_loss + (self.ub_initial_loss - self.lb_initial_loss) * \
                            torch.sigmoid(self.log_initial_loss)
        # self.growth_rate = 0.0045
        # self.carrying_capacity = torch.tensor(1.)
        # self.initial_loss = 0.0351
        # self.growth_rate = torch.exp(-self.log_growth_rate)
        # self.carrying_capacity = self.lb_carrying_capacity + (self.ub_carrying_capacity - self.lb_carrying_capacity) / 2 \
        #                          + (self.ub_carrying_capacity - self.lb_carrying_capacity) / 2 \
        #                          * torch.tanh(self.log_carrying_capacity)
        # self.initial_loss = self.lb_initial_loss + (self.ub_initial_loss - self.lb_initial_loss) / 2 \
        #                          + (self.ub_initial_loss - self.lb_initial_loss) / 2 \
        #                          * torch.tanh(self.log_initial_loss)

        s = inputs[:, :, 0: self.inputs_dim - 1]
        t = inputs[:, :, self.inputs_dim - 1:]
        s.requires_grad_(True)
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])
        t_norm.requires_grad_(True)

        U_norm = self.neural_net(x=torch.cat((s_norm, t_norm), dim=2))
        # U_norm = self.neural_net(t_norm)

        U = inverse_standardize_tensor(U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])

        grad_outputs = torch.ones_like(U)
        U_t = torch.autograd.grad(
            U, t_norm,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        U_tt = torch.autograd.grad(
            U_t, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        U_s = torch.autograd.grad(
            U, s,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # # F = dU_dt_pred - Verhulst(y=U, r=self.growth_rate, K=self.carrying_capacity, C=self.initial_loss)
        # Verhulst_pred = Verhulst(y=U, r=self.growth_rate, K=self.carrying_capacity, C=self.initial_loss)
        # tolerance = 1e-8 * torch.ones_like(dU_dt_pred)
        # F = (dU_dt_pred - Verhulst_pred) / torch.where(torch.abs(Verhulst_pred) > tolerance, Verhulst_pred,
        #                                                     tolerance)

        # G = self.dynamicNN(x=U_norm)
        # G = self.dynamicNN(x=torch.cat((s_norm, t_norm, U_norm), dim=2))
        G = Verhulst(y=U, r=self.growth_rate, K=self.carrying_capacity, C=self.initial_loss)
        G_t = torch.autograd.grad(
            G, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        F = U_t - G
        F_t = U_tt - G_t
        # self.tolerance = 1e-8 * torch.ones_like(U_t)
        # F = (U_t - G) / torch.where(torch.abs(U_t) > self.tolerance, U_t, self.tolerance)
        # F_t = (U_tt - G_t) / torch.where(torch.abs(U_tt) > self.tolerance, U_tt, self.tolerance)

        self.U_t = U_t
        return U, F, F_t


class RULPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        super(RULPINN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        self.neural_net = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

        self.dynamicNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim + 1,
            outputs_dim=1,
            layers=layers
        )

    def forward(self, inputs):
        s = inputs[:, :, 0: self.inputs_dim - 1]
        t = inputs[:, :, self.inputs_dim - 1:]
        s.requires_grad_(True)
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][0: self.inputs_dim - 1],
                                          std=self.scaler_inputs[1][0: self.inputs_dim - 1])
        t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][self.inputs_dim - 1:],
                                          std=self.scaler_inputs[1][self.inputs_dim - 1:])

        U_norm = self.neural_net(x=torch.cat((s_norm, t_norm), dim=2))
        # U_norm = self.neural_net(t_norm)

        U = inverse_standardize_tensor(U_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])
        # U = U_norm

        grad_outputs = torch.ones_like(U)
        U_t = torch.autograd.grad(
            U, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        U_s = torch.autograd.grad(
            U, s,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        G = self.dynamicNN(x=torch.cat((s_norm, t_norm, U_norm), dim=2))

        F = U_t - G
        F_t = torch.zeros_like(U)
        # self.tolerance = 1e-8 * torch.ones_like(U_t)
        # F = (U_t - G) / torch.where(torch.abs(U_t) > self.tolerance, U_t, self.tolerance)
        # F_t = (U_tt - G_t) / torch.where(torch.abs(U_tt) > self.tolerance, U_tt, self.tolerance)

        return U, F, F_t


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs_U, targets_U, outputs_F, outputs_F_t, log_var_u, log_var_f):
        # loss_h = torch.sqrt(torch.mean((outputs_h - targets_h) ** 2))
        # loss_y = torch.sqrt(torch.mean((outputs_y - targets_y)**2))
        # loss_e = torch.sqrt(torch.mean((outputs_e)**2))
        # loss = 10*loss_h + loss_y + loss_e

        # loss_U = torch.sqrt(torch.mean((outputs_U - targets_U) ** 2))
        # loss_U = torch.mean(torch.abs((outputs_U - targets_U) / targets_U))
        # loss_U = torch.sqrt(torch.mean((outputs_U - targets_U) ** 2))
        # loss_U = torch.sqrt(torch.mean((log_var_u * (outputs_U - targets_U)) ** 2))
        loss_U = torch.sum((outputs_U - targets_U) ** 2)

        # loss_F = torch.sqrt(torch.mean((outputs_F) ** 2))
        # loss_F = torch.mean(torch.abs(outputs_F))
        loss_F = torch.sum(outputs_F ** 2)

        loss_F_t = torch.sum((outputs_F_t) ** 2)
        # loss_F_t = torch.mean(torch.abs(outputs_F_t))

        # loss = loss_U / loss_U.detach() + loss_F / loss_F.detach()
        # loss = 20*max(loss_U - 0.0058, 0) + loss_F
        # loss = loss_U + 1e-1*loss_F
        loss = torch.exp(-log_var_u) * loss_U + torch.exp(-log_var_f) * loss_F + log_var_u + log_var_f
        # print(' Loss_U: {:.5f}, Loss_F: {:.5f},'.format(loss_U, loss_F))

        self.loss_U = loss_U
        self.loss_F = loss_F
        self.loss_F_t = loss_F_t
        return loss


def train(num_epoch, batch_size, train_loader, num_slices_train, inputs_val, targets_val,
          model, optimizer, scheduler, criterion, log_sigma_u, log_sigma_f):
    num_period = int(num_slices_train / batch_size)
    loss_epoch_train = torch.zeros(num_epoch)
    loss_epoch_val = torch.zeros(num_epoch)
    growth_rate_epoch = torch.zeros(num_epoch)
    carrying_capacity_epoch = torch.zeros(num_epoch)
    initial_loss_epoch = torch.zeros(num_epoch)
    model.train()
    for epoch in range(num_epoch):
        loss_period_train = torch.zeros(num_period)
        # growth_rate_period = torch.zeros(num_period)
        # carrying_capacity_period = torch.zeros(num_period)
        # initial_loss_period = torch.zeros(num_period)
        with torch.backends.cudnn.flags(enabled=False):
            for period, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
                log_var_u = log_sigma_u
                log_var_f = log_sigma_f
                optimizer.zero_grad()
                U_pred_train, F_pred_train, F_t_pred_train = model(inputs=inputs_train_batch)
                loss = criterion(
                    outputs_U=U_pred_train,
                    targets_U=targets_train_batch,
                    outputs_F=F_pred_train,
                    outputs_F_t=F_t_pred_train,
                    log_var_u=log_var_u,
                    log_var_f=log_var_f
                )
                loss.backward()
                optimizer.step()
                loss_period_train[period] = criterion.loss_U.detach()
                # growth_rate_period[period] = model.growth_rate.detach()
                # carrying_capacity_period[period] = model.carrying_capacity.detach()
                # initial_loss_period[period] = model.initial_loss.detach()

                if (epoch + 1) % 1 == 0 and (period + 1) % 1 == 0:  # 每 100 次输出结果
                    print(
                        'Epoch: {}, Period: {}, Loss: {:.5f}, Loss_U: {:.5f}, Loss_F: {:.5f}, Loss_F_t: {:.5f}'.format(
                            epoch + 1, period + 1, loss, criterion.loss_U, criterion.loss_F, criterion.loss_F_t))

        loss_epoch_train[epoch] = torch.mean(loss_period_train)
        # growth_rate_epoch[epoch] = torch.mean(growth_rate_period)
        # carrying_capacity_epoch[epoch] = torch.mean(carrying_capacity_period)
        # initial_loss_epoch[epoch] = torch.mean(initial_loss_period)

        model.eval()
        U_pred_val, F_pred_val, F_t_pred_val = model(inputs=inputs_val)
        loss_val = criterion(
            outputs_U=U_pred_val,
            targets_U=targets_val,
            outputs_F=F_pred_val,
            outputs_F_t=F_t_pred_val,
            log_var_u=log_var_u,
            log_var_f=log_var_f
        )
        scheduler.step()
        loss_epoch_val[epoch] = criterion.loss_U.detach()

    return model, loss_epoch_train, loss_epoch_val, growth_rate_epoch, carrying_capacity_epoch, initial_loss_epoch


pass