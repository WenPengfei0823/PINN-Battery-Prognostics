# PINN_Battery_Prognostics

This repository includes the code and data for the paper "***Fusing Models for Prognostics and Health Management of Lithium-Ion Batteries Based on Physics-Informed Neural Networks***"

## Abstract

For Prognostics and Health Management (PHM) of Lithium-ion (Li-ion) batteries, many models have been established to characterize their degradation process. The existing empirical or physical models can reveal important information regarding the degradation dynamics. However, there is no general and flexible methods to fuse the information represented by those models. Physics-Informed Neural Network (PINN) is an efficient tool to fuse empirical or physical dynamic models with data-driven models. To make full use of various information sources, we propose a model fusion scheme based on PINN. This is done by developing a semi-empirical semi-physical Partial Differential Equation (PDE) to model the dynamics of Li-ion batteries degrading. When there is little prior knowledge about the dynamics, we leverage the data-driven Deep Hidden Physics Model (DeepHPM) to discover the underlying governing dynamic models. Information on the dynamics is then fused with that mined by the surrogate neural network in the PINN framework. Moreover, an uncertainty-based adaptive weighting method is employed to balance the multiple learning tasks when training the PINN. The proposed methods are verified on a public cycling dataset of Li-ion Phosphate (LFP)/graphite batteries.

![PINN-Verhulst](https://github.com/WenPengfei0823/PINN-Battery-Prognostics/blob/main/Documents/PINN_Verhulst.jpg "Model fusion with *a priori* known dynamic model.")

![PINN-DeepHPM](https://github.com/WenPengfei0823/PINN-Battery-Prognostics/blob/main/Documents/PINN_DeepHPM.jpg "Model fusion without *a priori* known dynamic model.")

## Citation

> ```
> @article{Wen2023_PINN,
>    author = {Wen, Pengfei and Ye, Zhi-Sheng and Li, Yong and Chen, Shaowei and Zhao, Shuai},
>    title = {Fusing Models for Prognostics and Health Management of Lithium-Ion Batteries Based on Physics-Informed Neural Networks},
>    journal = {arXiv preprint: },
>    year = {2023},
>    type = {Journal Article},   
>    doi = {},
> }
> ```
