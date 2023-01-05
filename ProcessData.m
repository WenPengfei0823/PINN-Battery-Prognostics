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
    %  Remove the empty cycles at first cycle.
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
    
    %  Remove the cycles that the corresponding RUL is negative.
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

save('SeversonBattery.mat','Features_mov_Flt', 'RUL_Flt', 'Cycles_Flt', ...
    'PCL_Flt', 'Num_Cycles_Flt', 'train_ind', 'test_ind', 'secondary_test_ind');