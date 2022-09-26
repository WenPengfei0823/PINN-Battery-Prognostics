import numpy as np
import torch
import matplotlib.pyplot as plt

results_SoH_CaseA_Baseline = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseA_Baseline.pth')
results_SoH_CaseA_Verhulst_Sum = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseA_Verhulst_Sum.pth')
results_SoH_CaseA_Verhulst_AdpBal = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseA_Verhulst_AdpBal.pth')
results_SoH_CaseA_DeepHPM_Sum = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseA_DeepHPM_Sum.pth')
results_SoH_CaseA_DeepHPM_AdpBal = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseA_DeepHPM_AdpBal.pth')

plt.figure()
plt.plot(
    results_SoH_CaseA_Baseline['Cycles'],
    results_SoH_CaseA_Baseline['U_true'],
    linewidth=2,
    color=(238/255, 28/255, 37/255),
    label='Ground Truth'
)
plt.plot(
    results_SoH_CaseA_Baseline['Cycles'],
    results_SoH_CaseA_Baseline['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(238/255, 28/255, 37/255),
    alpha=1.,
    label='Baseline'
)
plt.plot(
    results_SoH_CaseA_Verhulst_Sum['Cycles'],
    results_SoH_CaseA_Verhulst_Sum['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(1/255, 168/255, 158/255),
    alpha=1.,
    label='PINN-Verhulst (Sum)'
)
plt.plot(
    results_SoH_CaseA_Verhulst_AdpBal['Cycles'],
    results_SoH_CaseA_Verhulst_AdpBal['U_pred'],
    linewidth=2,
    color=(1/255, 168/255, 158/255),
    alpha=1.,
    label='PINN-Verhulst (AdpBal)'
)
plt.plot(
    results_SoH_CaseA_DeepHPM_Sum['Cycles'],
    results_SoH_CaseA_DeepHPM_Sum['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(0/255, 84/255, 165/255),
    alpha=1.,
    label='PINN-DeepHPM (Sum)'
)
plt.plot(
    results_SoH_CaseA_DeepHPM_AdpBal['Cycles'],
    results_SoH_CaseA_DeepHPM_AdpBal['U_pred'],
    linewidth=2,
    color=(0/255, 84/255, 165/255),
    alpha=1.,
    label='PINN-DeepHPM (AdpBal)'
)
plt.legend()
plt.xlabel('Monitoring Time (Unit: Cycles)')
plt.ylabel('State of Health (Unit: 1)')
plt.title('SoH Estimation Cell #124')
plt.show()

results_SoH_CaseB_Baseline = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseB_Baseline.pth')
results_SoH_CaseB_Verhulst_Sum = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseB_Verhulst_Sum.pth')
results_SoH_CaseB_Verhulst_AdpBal = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseB_Verhulst_AdpBal.pth')
results_SoH_CaseB_DeepHPM_Sum = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseB_DeepHPM_Sum.pth')
results_SoH_CaseB_DeepHPM_AdpBal = torch.load('..\Results\\4 Presentation\SoH Estimation\SoH_CaseB_DeepHPM_AdpBal.pth')

plt.figure()
plt.plot(
    results_SoH_CaseB_Baseline['Cycles'],
    results_SoH_CaseB_Baseline['U_true'],
    linewidth=2,
    color=(238/255, 28/255, 37/255),
    label='Ground Truth'
)
plt.plot(
    results_SoH_CaseB_Baseline['Cycles'],
    results_SoH_CaseB_Baseline['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(238/255, 28/255, 37/255),
    alpha=1.,
    label='Baseline'
)
plt.plot(
    results_SoH_CaseB_Verhulst_Sum['Cycles'],
    results_SoH_CaseB_Verhulst_Sum['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(1/255, 168/255, 158/255),
    alpha=1.,
    label='PINN-Verhulst (Sum)'
)
plt.plot(
    results_SoH_CaseB_Verhulst_AdpBal['Cycles'],
    results_SoH_CaseB_Verhulst_AdpBal['U_pred'],
    linewidth=2,
    color=(1/255, 168/255, 158/255),
    alpha=1.,
    label='PINN-Verhulst (AdpBal)'
)
plt.plot(
    results_SoH_CaseB_DeepHPM_Sum['Cycles'],
    results_SoH_CaseB_DeepHPM_Sum['U_pred'],
    linestyle='--',
    linewidth=2,
    color=(0/255, 84/255, 165/255),
    alpha=1.,
    label='PINN-DeepHPM (Sum)'
)
plt.plot(
    results_SoH_CaseB_DeepHPM_AdpBal['Cycles'],
    results_SoH_CaseB_DeepHPM_AdpBal['U_pred'],
    linewidth=2,
    color=(0/255, 84/255, 165/255),
    alpha=1.,
    label='PINN-DeepHPM (AdpBal)'
)
plt.legend()
plt.xlabel('Monitoring Time (Unit: Cycles)')
plt.ylabel('State of Health (Unit: 1)')
plt.title('SoH Estimation Cell #116')
plt.show()