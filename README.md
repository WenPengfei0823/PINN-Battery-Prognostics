# PINN_Battery_Prognostics

This repository includes the code and data for the paper "***Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries***"

## Abstract

For Prognostics and Health Management (PHM) of Lithium-ion (Li-ion) batteries, many models have been established to characterize their degradation process. The existing empirical or physical models can reveal important information regarding the degradation dynamics. However, there are no general and flexible methods to fuse the information represented by those models. Physics-Informed Neural Network (PINN) is an efficient tool to fuse empirical or physical dynamic models with data-driven models. To take full advantage of various information sources, we propose a model fusion scheme based on PINN. It is implemented by developing a semi-empirical semi-physical Partial Differential Equation (PDE) to model the degradation dynamics of Li-ion batteries. When there is little prior knowledge about the dynamics, we leverage the data-driven Deep Hidden Physics Model (DeepHPM) to discover the underlying governing dynamic models. The uncovered dynamics information is then fused with that mined by the surrogate neural network in the PINN framework. Moreover, an uncertainty-based adaptive weighting method is employed to balance the multiple learning tasks when training the PINN. The proposed methods are verified on a public dataset of Li-ion Phosphate (LFP)/graphite batteries.

![PINN-Verhulst](https://github.com/WenPengfei0823/PINN-Battery-Prognostics/blob/main/Documents/PINN_Verhulst.jpg "Model fusion with *a priori* known dynamic model.")

![PINN-DeepHPM](https://github.com/WenPengfei0823/PINN-Battery-Prognostics/blob/main/Documents/PINN_DeepHPM.jpg "Model fusion without *a priori* known dynamic model.")

## Citation

> ```
> @article{Wen2023_PINN,
> author = {Wen, Pengfei and Ye, Zhi-Sheng and Li, Yong and Chen, Shaowei and Xie, Pu and Zhao, Shuai},
> title = {Physics-informed neural networks for prognostics and health management of lithium-ion batteries},
> journal = {IEEE Transactions on Intelligent Vehicles},
> year = {2024},
> type = {Journal Article},
> volume = {9},
> number = {1},
> pages = {2276-2289},
> doi = {10.1109/TIV.2023.3315548}
> }
> ```


## Instruction

1. Run "LoadData.m" to load data.
2. Run "ProcessData.m" to process data.
3. Run "ExtractFeature.m" to extract the involved features.
4. Then the users can run the codes in the folders "Experiments" to replicate our experiments.
