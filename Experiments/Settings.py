import torch

num_rounds = 5

settings_SoH_CaseA = dict()
settings_SoH_CaseA['num_rounds'] = num_rounds
settings_SoH_CaseA['batch_size'] = 1024
settings_SoH_CaseA['num_epoch'] = 2000
settings_SoH_CaseA['num_layers'] = [2]
settings_SoH_CaseA['num_neurons'] = [128]
settings_SoH_CaseA['lr'] = 1e-3
settings_SoH_CaseA['step_size'] = 50000
settings_SoH_CaseA['gamma'] = 0.1
settings_SoH_CaseA['inputs_lib_dynamical'] = ['s_norm, t_norm']
settings_SoH_CaseA['inputs_dim_lib_dynamical'] = ['inputs_dim']
torch.save(settings_SoH_CaseA, 'Settings\\settings_SoH_CaseA.pth')

settings_SoH_CaseB = dict()
settings_SoH_CaseB['num_rounds'] = num_rounds
settings_SoH_CaseB['batch_size'] = 1024
settings_SoH_CaseB['num_epoch'] = 2000
settings_SoH_CaseB['num_layers'] = [2]
settings_SoH_CaseB['num_neurons'] = [64]
settings_SoH_CaseB['lr'] = 1e-3
settings_SoH_CaseB['step_size'] = 50000
settings_SoH_CaseB['gamma'] = 0.1
settings_SoH_CaseB['inputs_lib_dynamical'] = ['t_norm']
settings_SoH_CaseB['inputs_dim_lib_dynamical'] = ['1']
torch.save(settings_SoH_CaseB, 'Settings\\settings_SoH_CaseB.pth')

settings_RUL_CaseA = dict()
settings_RUL_CaseA['num_rounds'] = num_rounds
settings_RUL_CaseA['batch_size'] = 1024
settings_RUL_CaseA['num_epoch'] = 2000
settings_RUL_CaseA['num_layers'] = [2]
settings_RUL_CaseA['num_neurons'] = [128]
settings_RUL_CaseA['lr'] = 1e-3
settings_RUL_CaseA['step_size'] = 50000
settings_RUL_CaseA['gamma'] = 0.1
settings_RUL_CaseA['inputs_lib_dynamical'] = ['s_norm, t_norm, U_norm']
settings_RUL_CaseA['inputs_dim_lib_dynamical'] = ['inputs_dim + 1']
torch.save(settings_RUL_CaseA, 'Settings\\settings_RUL_CaseA.pth')

settings_RUL_CaseB = dict()
settings_RUL_CaseB['num_rounds'] = num_rounds
settings_RUL_CaseB['batch_size'] = 1024
settings_RUL_CaseB['num_epoch'] = 2000
settings_RUL_CaseB['num_layers'] = [2]
settings_RUL_CaseB['num_neurons'] = [128]
settings_RUL_CaseB['lr'] = 1e-3
settings_RUL_CaseB['step_size'] = 50000
settings_RUL_CaseB['gamma'] = 0.1
settings_RUL_CaseB['inputs_lib_dynamical'] = ['t_norm, U_norm, U_s']
settings_RUL_CaseB['inputs_dim_lib_dynamical'] = ['inputs_dim + 1']
torch.save(settings_RUL_CaseB, 'Settings\\settings_RUL_CaseB.pth')

settings_RUL_CaseC = dict()
settings_RUL_CaseC['num_rounds'] = num_rounds
settings_RUL_CaseC['batch_size'] = 8192
settings_RUL_CaseC['num_epoch'] = 8000
settings_RUL_CaseC['num_layers'] = [4]
settings_RUL_CaseC['num_neurons'] = [128]
settings_RUL_CaseC['lr'] = 1e-3
settings_RUL_CaseC['step_size'] = 50000
settings_RUL_CaseC['gamma'] = 0.1
settings_RUL_CaseC['inputs_lib_dynamical'] = ['t_norm']
settings_RUL_CaseC['inputs_dim_lib_dynamical'] = ['1']
torch.save(settings_RUL_CaseC, 'Settings\\settings_RUL_CaseC.pth')