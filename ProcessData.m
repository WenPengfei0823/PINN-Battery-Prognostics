cap_nom = 1.1;
unit_num = length(batch_combined);
Cycles = cell(unit_num, 1);
Tavg = cell(unit_num, 1);
Qd = cell(unit_num, 1);
Tavg_all = zeros(unit_num, 1);
SoH = cell(unit_num, 1);
PCL = cell(unit_num, 1);
Features = cell(unit_num, 1);
RUL = cell(unit_num, 1);

Features_Flt = [];
SoH_Flt = [];
PCL_Flt = [];
Cycles_Flt = [];
RUL_Flt = [];
Num_Cycles_Flt = zeros(unit_num, 1);

Name_Features = [];


for i = 1:unit_num,
    %  remove the empty cycles at first cycle
    if isempty(batch_combined(i).cycles(1).Qd),
        batch_combined(i).cycles(1) = [];
        batch_combined(i).cycle_life = batch_combined(i).cycle_life;
        batch_combined(i).summary.cycle(1) = [];
        batch_combined(i).summary.QDischarge(1) = [];
        batch_combined(i).summary.QCharge(1) = [];
        batch_combined(i).summary.IR(1) = [];
        batch_combined(i).summary.Tmax(1) = [];
        batch_combined(i).summary.Tavg(1) = [];
        batch_combined(i).summary.Tmin(1) = [];
        batch_combined(i).summary.chargetime(1) = [];
    end
    
    %  remove the cycles that the corresponding RUL is negative
    if batch_combined(i).cycle_life < length(batch_combined(i).cycles),
        batch_combined(i).cycles(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.cycle(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.QDischarge(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.QCharge(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.IR(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.Tmax(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.Tavg(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.Tmin(batch_combined(i).cycle_life+1:end) = [];
        batch_combined(i).summary.chargetime(batch_combined(i).cycle_life+1:end) = [];
    end    
    
    Cycles_i = batch_combined(i).summary.cycle;
    Tavg_i = batch_combined(i).summary.Tavg;
    Qd_i = batch_combined(i).summary.QDischarge;
    N_i = length(batch_combined(i).cycles);
    avg_Qd_i = mean(Qd_i);
    std_Qd_i = std(Qd_i);
    for j = 1:N_i,
        if (Qd_i(j) <= avg_Qd_i - 3 * std_Qd_i) || ...
                (Qd_i(j) >= avg_Qd_i + 3 * std_Qd_i),
            if j < 2,
                Qd_i(j) = Qd_i(j+1);
            elseif j > N_i-1,
                Qd_i(j) = Qd_i(j-1);
            else
                Qd_i(j) = (Qd_i(j-1) + Qd_i(j+1)) / 2;
            end
        end
    end
    
    Features_i = ExtractFeature(batch_combined(i));
%     Features_i = [batch_combined(i).summary.IR, batch_combined(i).summary.QDischarge, ...
%         batch_combined(i).summary.Tavg, batch_combined(i).summary.chargetime];
    RUL_i = batch_combined(i).cycle_life - Cycles_i;
    
    Features{i} = Features_i;
    Cycles{i} = Cycles_i;
    Tavg{i} = Tavg_i;
    Qd{i} = Qd_i;
    Tavg_all(i) = mean(Tavg_i);
    SoH{i} = Qd_i / cap_nom;
    PCL{i} = 1 - SoH{i};
    RUL{i} = RUL_i;
    display(['Processing cell #', num2str(i)]);
    
    Features_Flt = [Features_Flt; Features{i}];
    SoH_Flt = [SoH_Flt; SoH{i}];
    PCL_Flt = [PCL_Flt; PCL{i}];
    Cycles_Flt = [Cycles_Flt; Cycles{i}];
    RUL_Flt = [RUL_Flt; RUL{i}];
    Num_Cycles_Flt(i) = N_i;
end

Features_mov_Flt = [];
Features_mov = cell(unit_num, 1);
windowSize = 10;
for i = 1: unit_num,
    Features_mov{i} = zeros(size(Features{i}));
    for k = 1:windowSize-1,
        Features_mov{i}(k, :) = mean(Features{i}(1:k, :), 1);
    end
    ma = tsmovavg(Features{i}, 's', windowSize, 1);
    Features_mov{i}(windowSize:end, :) = ma(windowSize:end, :);
    Features_mov_Flt = [Features_mov_Flt; Features_mov{i}];
end
% clear batch_combined
save('SeversonBattery.mat','Features_mov_Flt', 'RUL_Flt', 'Cycles_Flt', ...
    'PCL_Flt', 'Num_Cycles_Flt', 'train_ind', 'test_ind', 'secondary_test_ind');
%%
[rsquare_v_order, SSE_v_order] = findFitOrder(Cycles, SoH);

SoH_fit = cell(unit_num, 1);
p = 5;
ft = ['poly', num2str(p)];
for i = 1: unit_num,
    [fitresult, ~] = createFit(Cycles{i}, SoH{i}, ft);
    SoH_fit{i} = fitresult(Cycles{i});
end
%%
% max_wdsz = 20;
% PDE_rsquare_v_wdsz = zeros(unit_num, max_wdsz/2);
% for w = 2:2:max_wdsz,
SoH_mov = cell(unit_num, 1);
windowSize = 10;
for i = 1: unit_num,
    ma = tsmovavg(SoH{i}, 'e', windowSize, 1);
    SoH_mov{i} = [SoH{i}(1: windowSize-1); ma(windowSize: end)];
end

p_PDE = 2;
PDE_rsquare = zeros(unit_num, 1);
PDE_params = zeros(unit_num, p_PDE+1);
Verhulst_params = zeros(unit_num, 2);
ft = ['poly1'];
y_all = [];
dy_dt_all = [];
for i = 1: unit_num,
    y = 1-SoH_mov{i};
    dy_dt = diff(y);
    y = y(2:end);
    [fitresults, gof] = createFitVerhulst(y, dy_dt);
    PDE_rsquare(i) = gof.rsquare;
    Verhulst_params(i, 1) = fitresults.r;
    Verhulst_params(i, 2) = fitresults.K;
    
    y_all = [y_all; y];
    dy_dt_all = [dy_dt_all; dy_dt];
%     [~, gof] = createFit(y, dy_dt, ft);
%     PDE_rsquare(i) = gof.rsquare;
%     
%     [PDE_params_i,~] = polyfit(y, dy_dt, p_PDE);
%     PDE_params(i, :) = PDE_params_i;
end
%     display(['windowSize = ', num2str(w)])
%     PDE_rsquare_v_wdsz(:, w/2) = PDE_rsquare;
% end
%%
PCL_norm = cell(size(PCL));
PCL_pred = cell(size(PCL));
PCL_pred_Flt = [];
% p_all = zeros(length(train_ind), 3);
p_all = zeros(numBat, 3);
for i = 1:numBat,
PCL_norm{i} = (PCL{i} - min(PCL{i})) ./ (max(PCL{i}) - min(PCL{i}));
% idx_cell = train_ind(i);
idx_cell = i;
t_start = 1;
t_end = length(SoH{idx_cell});
% t_end = 300;
y = PCL{idx_cell}(t_start: t_end);
y0 = y(1);
dy_dt = diff(y);
tspan = t_start+1: t_end;
y_true = y(2:end);

p_0 = [1e-3, 0.2, y0];
A = [];
b = [];    
Aeq = [];
beq = [];
Lb = [0, 0.2, 0];
Ub = [Inf, 1, y0];  
nonlcon = [];
options = optimoptions('fmincon', 'Algorithm', 'sqp');
[p, fVal,Flag] = fmincon(@(p) FitOde(p, y0, tspan, y_true), p_0, A, b, Aeq, beq, Lb, Ub, nonlcon, options);
p_all(i, :) = p;

[~, y_pred] = ode45(@(t,y) p(1) * (y-p(3)) * (1 - (y-(p(3))) / (p(2)-p(3))), tspan, y0);
PCL_pred{i} = [y0; y_pred];
PCL_pred_Flt = [PCL_pred_Flt; PCL_pred{i}];
display(['Processing cell #', num2str(i)]);
end
%%
idx_cell = 116;
% p=[0.0037, 0.9996, 0.0348];
% p=[0.002, 0.4524, 0.0207];
p = p_all(idx_cell, :);
t_end = length(SoH{idx_cell});
y = PCL{idx_cell}(t_start:t_end);
y0 = y(1);
dy_dt = diff(y);
tspan = t_start+1: t_end;
y_true = y(2:end);
% p = [0.0003, 1, 0.0134];
[~, y_pred] = ode45(@(t,y) p(1) * (y-p(3)) * (1 - (y-(p(3))) / (p(2)-p(3))), tspan, y0);
% [~, y_pred] = ode45(@(t,y) p(1) * (y) * (1 - (y) / (p(2))), tspan, y0);
%%%%%% one-step prediction %%%%%%
% seq_len = 101;
% y_pred = zeros(size(y_true));
% y_pred(1) = y_true(1);
% for k = seq_len:length(y_true),
%     [~, y_pred_tmp] = ode45(@(t,y) p(1) * (y-p(3)) * (1 - (y-(p(3))) / (p(2)-p(3))), [tspan(k-(seq_len-1)), tspan(k)], y_true(k-(seq_len-1)));
%     y_pred(k) = y_pred_tmp(end);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on
scatter(tspan, 1.1-1.1*y_true, '.');
plot(tspan, 1.1-1.1*y_pred, 'LineWidth', 2);
xlabel('Cycles')
ylabel('Capacity / Ah')
title(['Cell #', num2str(idx_cell)])
hold off
%%
Cycles_train = Cycles(train_ind);
SoH_train = SoH(train_ind);
PCL_train = cell(size(SoH_train));
y0_train = zeros(size(SoH_train));
for i = 1:length(SoH_train),
    PCL_train{i} = 1 - SoH_train{i};
    y0_train(i) = PCL_train{i}(1);
end
p_0 = [1e-3 0.2 min(y0_train)];
A = [];
b = [];    
Aeq = [];
beq = [];
Lb = [0, 0.2, 0];
Ub = [1e-2, 1, min(y0_train)];  
nonlcon = [];
options = optimoptions('fmincon', 'Algorithm', 'sqp');
global iter;
iter = 1;
[p, fVal,Flag] = fmincon(@(p) FitOdeAll(p, y0_train, Cycles_train, PCL_train), p_0, A, b, Aeq, beq, Lb, Ub, nonlcon, options);
%%
% lambda = 0.5;
% X0 = SoH;
% X0_pred = cell(unit_num, 1);
% X1 = cell(unit_num, 1);
% X1_pred = cell(unit_num, 1);
% Z1 = cell(unit_num, 1);
% PAR = cell(unit_num, 1);
% for i = 1: unit_num,
%     X1{i} = cumsum(X0{i});
%     Z1{i} = lambda * X1{i}(2:end) + (1-lambda) * X1{i}(1:end-1);
%     B = [-Z1{i}, Z1{i}.^2];
%     Y = X0{i}(2:end);
%     PAR{i} = pinv(B' * B) * B' * Y;
%     a = PAR{i}(1);
%     b = PAR{i}(2);
%     x_init = X0{i}(1);
%     X1_pred{i} = a*x_init ./ (b*x_init + (a - b*x_init).*exp(a*t{i}));
%     X0_pred{i} = diff(X1_pred{i});
% end
%%
% lambda = 0.5;
% X1 = SoH;
% X0 = cell(unit_num, 1);
% X1_pred = cell(unit_num, 1);
% Z1 = cell(unit_num, 1);
% PAR = cell(unit_num, 1);
% for i = 1: unit_num,
%     X0{i} = [X1{i}(1); diff(X1{i})];
%     Z1{i} = [X1{i}(1); lambda * X1{i}(2:end) + (1-lambda) * X1{i}(1:end-1)];
%     B = [-Z1{i}(2:end), (Z1{i}(2:end)).^2];
%     Y = X0{i}(2:end);
%     PAR{i} = pinv(B' * B) * B' * Y;
%     a = PAR{i}(1);
%     b = PAR{i}(2);
%     x_init = X1{i}(1);
%     X1_pred{i} = a*x_init ./ (b*x_init + (a - b*x_init)*exp(a*t{i}));
% end
%% Liu et al. (2020)
% CCCT_p1=zeros(1850, 1);
% for i = 2:1851,
% idx_CC = (batch_combined(1).cycles(i).I<1.001) & (batch_combined(1).cycles(i).I>0.999);
% CCCT_i = max(batch_combined(1).cycles(i).t(idx_CC)) - min(batch_combined(1).cycles(i).t(idx_CC));
%     if ~isempty(CCCT_i),
%         CCCT_p1(i-1) = CCCT_i;
%     else
%         CCCT_p1(i-1) = nan;
%     end
% end
% for j = 1+1:1850-1,
%     if (CCCT_p1(j) >= mean(CCCT_p1(1:j-1)) + 3 * std(CCCT_p1(1:j-1))) || (CCCT_p1(j-1) <= mean(CCCT_p1(1:j-1)) - 3 * std(CCCT_p1(1:j))),
%         CCCT_p1(j) = (CCCT_p1(j+1) + CCCT_p1(j-1)) / 2;
%     end
% end
% %% Xu et al. (2018)
% clear coeff_emp;
% coeff_emp.a = zeros(unit_num, 1);
% coeff_emp.b = zeros(unit_num, 1);
% coeff_emp.c = zeros(unit_num, 1);
% coeff_emp.f_d_1 = zeros(unit_num, 1);
% 
% num_coeff = 4;
% coeff_est_flag = zeros(unit_num, 1);
% coeff_0 = ones(num_coeff, 1) * 1e-3;
% goals = 0;
% weights = 1;
% A = [];
% b = [];    
% Aeq = [];
% beq = [];
% Lb = zeros(num_coeff, 1);
% Ub = ones(num_coeff, 1);     
% nonlcon = [];
% options = optimoptions('fmincon', 'Algorithm', 'sqp', 'MaxIter', 20200);
% %%%%%%%%%%%%%单目标优化%%%%%%%%%%%%
% for i = 1:unit_num,
%     MSE = @(x) mean(((x(1)*exp(x(2)*x(4)*t{i}) + x(3)*exp(x(4)*t{i})) - ...
%         (1 - Qd{i}/1.1)).^2);
%     [x, fVal, flag] = fmincon(MSE, coeff_0, A, b, Aeq, beq, Lb, Ub, nonlcon, options);
%     coeff_est_flag(i) = flag;    
%     coeff_emp.a(i) = x(1);
%     coeff_emp.b(i) = x(2);
%     coeff_emp.c(i) = x(3);
%     coeff_emp.f_d_1(i) = x(4);
% end
% T_ref = 30;
% [~, T_ref_idx] = min((Tavg_all-T_ref).^2);
% f_d_1_ref = coeff_emp.f_d_1(T_ref_idx);
% f_d_1_norm = coeff_emp.f_d_1 / f_d_1_ref;