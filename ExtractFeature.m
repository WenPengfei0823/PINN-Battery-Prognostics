function Features_i = ExtractFeatures(batch_combined_i)
num_cycles = length(batch_combined_i.cycles);
V_E = zeros(num_cycles, 1);
T_E = zeros(num_cycles, 1);
V_FI = zeros(num_cycles, 1);
T_FI = zeros(num_cycles, 1);
V_CI = zeros(num_cycles, 1);
T_CI = zeros(num_cycles, 1);
V_CCI = zeros(num_cycles, 1);
T_CCI = zeros(num_cycles, 1);
V_SI = zeros(num_cycles, 1);
T_SI = zeros(num_cycles, 1);
V_KI = zeros(num_cycles, 1);
T_KI = zeros(num_cycles, 1);

dQdV_max = zeros(num_cycles, 1);
dQdV_min = zeros(num_cycles, 1);
dQdV_var = zeros(num_cycles, 1);
dQdV_ske = zeros(num_cycles, 1);
dQdV_kur = zeros(num_cycles, 1);

slope = zeros(num_cycles, 1);
intercept = zeros(num_cycles, 1);

for k = 1:num_cycles,
%     idx_discharge = (batch_combined_i.cycles(k).I < 0);
%     if isempty(batch_combined_i.cycles(k).discharge_dQdV),
%         batch_combined_i.cycles(k).discharge_dQdV = ...
%             diff(batch_combined_i.cycles(k).Qdlin) ./ ...
%             diff(batch_combined_i.Vdlin);
%     end
%     
%     idx_dQdV = (batch_combined_i.Vdlin > 2.7) & (batch_combined_i.Vdlin < 3.3);
%     
%     cycle_bench = 10;
%     dQdV_max(k) = max(batch_combined_i.cycles(k).Qdlin - ...
%         batch_combined_i.cycles(cycle_bench).Qdlin);
%     dQdV_min(k) = min(batch_combined_i.cycles(k).Qdlin - ...
%         batch_combined_i.cycles(cycle_bench).Qdlin);
%     dQdV_var(k) = var(batch_combined_i.cycles(k).Qdlin - ...
%         batch_combined_i.cycles(cycle_bench).Qdlin);
% %     dQdV_ske(k) = skewness(batch_combined_i.cycles(k).Qdlin - ...
% %         batch_combined_i.cycles(cycle_bench).Qdlin);
% %     dQdV_kur(k) = kurtosis(batch_combined_i.cycles(k).Qdlin - ...
% %         batch_combined_i.cycles(cycle_bench).Qdlin);

%%%%%%%%%%%%%%%%%%%%%%%%% Kong et al. 2021 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    windowSize = 10;
    dV = -0.0015;
    thd = 1e-3;
    dQdV_tmp = diff(batch_combined_i.cycles(k).Qdlin)./diff(batch_combined_i.Vdlin); % dQdV_discharge has outliers
    dQdV_tmp_ma = tsmovavg(dQdV_tmp, 's', windowSize, 1);
    dQ_tmp_ma = dQdV_tmp_ma * dV;
    N = length(dQ_tmp_ma);
    [~, idx_peak_dQ] = max(dQ_tmp_ma);
    for t = idx_peak_dQ:N,
        if dQ_tmp_ma(t) <= thd,
            idx_thd_dQ = t;
            y = dQ_tmp_ma(idx_peak_dQ:idx_thd_dQ);
            x = -batch_combined_i.cycles(k).Qdlin(idx_peak_dQ:idx_thd_dQ).^2;
            Y = y;
            X = [ones(idx_thd_dQ - idx_peak_dQ + 1, 1), x];
            p = pinv(X' * X) * X' * Y;
            slope(k) = p(1);
            intercept(k) = p(2);
            
            break
        end
    end
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
%     dQdV_max(k) = max(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
%     dQdV_min(k) = min(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
%     dQdV_var(k) = var(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
%     dQdV_ske(k) = skewness(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
%     dQdV_kur(k) = kurtosis(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
    
%     V_E(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).V(idx_discharge), 'Energy');
%     T_E(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).T(idx_discharge), 'Energy');
%     
%     V_FI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).V(idx_discharge), 'Fluctuation');
%     T_FI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).T(idx_discharge), 'Fluctuation');
%     
%     V_CI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).V(idx_discharge), 'Curvature');
%     T_CI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).T(idx_discharge), 'Curvature');
%     
% %     V_CCI = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
% %         batch_combined_i.cycles(k).V(idx_discharge), 'Concave');
% %     T_CCI = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
% %         batch_combined_i.cycles(k).T(idx_discharge), 'Concave');
%     
%     V_SI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).V(idx_discharge), 'Skewness');
%     T_SI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).T(idx_discharge), 'Skewness');
%     
%     V_KI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).V(idx_discharge), 'Kurtosis');
%     T_KI(k) = ComputeFeature(batch_combined_i.cycles(k).t(idx_discharge), ...
%         batch_combined_i.cycles(k).T(idx_discharge), 'Kurtosis');
end
% Features_i = [V_E, T_E, V_FI, T_FI, V_CI, T_CI, V_SI, T_SI, V_KI, T_KI];
% Features_i = [dQdV_max, dQdV_min, dQdV_var, dQdV_ske, dQdV_kur];
% Features_i = [dQdV_max, dQdV_min, dQdV_var];
Features_i = [slope, intercept, batch_combined_i.summary.Tavg];
end