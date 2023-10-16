function Features_i = ExtractFeatures(batch_combined_i)
num_cycles = length(batch_combined_i.cycles);

dQdV_max = zeros(num_cycles, 1);
dQdV_min = zeros(num_cycles, 1);
dQdV_var = zeros(num_cycles, 1);

slope = zeros(num_cycles, 1);
intercept = zeros(num_cycles, 1);

for k = 1:num_cycles,
    
    idx_dQdV = (batch_combined_i.Vdlin > 2.7) & (batch_combined_i.Vdlin < 3.3);

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
            
    dQdV_max(k) = max(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
    dQdV_min(k) = min(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
    dQdV_var(k) = var(batch_combined_i.cycles(k).discharge_dQdV(idx_dQdV));
    
end

Features_i = [slope, intercept, batch_combined_i.summary.Tavg, ...
    batch_combined_i.summary.IR, batch_combined_i.summary.chargetime, ...
    dQdV_max, dQdV_min, dQdV_var];
end