function T = Extract_Statistics_function(varargin)
%% start tic to display the total time requried at the end
tic
%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=",varargin{i+1},";"))
end

%% load the timeseries and delete EIS if exist
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.Current(1);
end

try
    TT = removevars(TT,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'});
end

%% now get the statistics
T_cell_name  = table(strings(1,1), 'VariableNames', {'Cell Name'});
T_time  = table(nan(1,1),nan(1,1), 'VariableNames', {'Start Date', 'Duration in days'});
T_capacity  = table(nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis'});
T_ah_throughput  = table(nan(1,1), 'VariableNames', {'Max'});
T_voltage  = table(nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis','Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance'});
T_current  = table(nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis','Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance'});
T_temperature  = table(nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1),nan(1,1), 'VariableNames', {'Max', 'Min', 'Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance','Histogram 5. Max Value','Histogram 5. Max Significance','Histogram 6. Max Value','Histogram 6. Max Significance','Histogram 7. Max Value','Histogram 7. Max Significance','Histogram 8. Max Value','Histogram 8. Max Significance','Histogram 9. Max Value','Histogram 9. Max Significance','Histogram 10. Max Value','Histogram 10. Max Significance'});


% Cell Name
T_cell_name.("Cell Name")(1)=cell_name;

% Time
try
    T_time.("Start Date")(1) = exceltime(min(TT.Time));
    T_time.("Duration in days")(1) = days(max(TT.Time)-min(TT.Time));
end


% Capacity
try
    T_capacity.Max(1) = max(TT.Capacity(not(isnan(TT.Capacity))));
    T_capacity.Min(1) = min(TT.Capacity(not(isnan(TT.Capacity))));
    T_capacity.Mean(1) = mean(TT.Capacity(not(isnan(TT.Capacity))));
    T_capacity.("Standard deviation")(1) = std(TT.Capacity(not(isnan(TT.Capacity))));
    T_capacity.Skewness(1) = skewness(TT.Capacity(not(isnan(TT.Capacity))));
    T_capacity.Kurtosis(1) = kurtosis(TT.Capacity(not(isnan(TT.Capacity))));
end

% Ah Throughput in Ah
try
    T_ah_throughput.Max(1) = max(TT.Ah_throughput(not(isnan(TT.Ah_throughput))));
end

% Voltage in V
try
    T_voltage.Max(1) = max(TT.Voltage(not(isnan(TT.Voltage))));
    T_voltage.Min(1) = min(TT.Voltage(not(isnan(TT.Voltage))));
    T_voltage.Mean(1) = mean(TT.Voltage(not(isnan(TT.Voltage))));
    T_voltage.("Standard deviation")(1) = std(TT.Voltage(not(isnan(TT.Voltage))));
    T_voltage.Skewness(1) = skewness(TT.Voltage(not(isnan(TT.Voltage))));
    T_voltage.Kurtosis(1) = kurtosis(TT.Voltage(not(isnan(TT.Voltage))));
end

try
    [voltage_N,voltage_edges] = histcounts(TT.Voltage(not(isnan(TT.Voltage))));
    voltage_edges = movmean(voltage_edges,2);
    voltage_edges = voltage_edges(2:end);
    voltage_N = voltage_N./sum(not(isnan(TT.Voltage(not(isnan(TT.Voltage))))));
    [voltage_pks,voltage_locs,voltage_w,voltage_p] = findpeaks(voltage_N);
    clear voltage_w
    [voltage_p,I] = sort(voltage_p,'descend');
    voltage_locs = voltage_locs(I);
    
    T_voltage.("Histogram 1. Max Value")(1) = voltage_edges(voltage_locs(1));
    T_voltage.("Histogram 1. Max Significance")(1) = voltage_N(voltage_locs(1));
    T_voltage.("Histogram 2. Max Value")(1) = voltage_edges(voltage_locs(2));
    T_voltage.("Histogram 2. Max Significance")(1) = voltage_N(voltage_locs(2));
    T_voltage.("Histogram 3. Max Value")(1) = voltage_edges(voltage_locs(3));
    T_voltage.("Histogram 3. Max Significance")(1) = voltage_N(voltage_locs(3));
    T_voltage.("Histogram 4. Max Value")(1) = voltage_edges(voltage_locs(4));
    T_voltage.("Histogram 4. Max Significance")(1) = voltage_N(voltage_locs(4));
end

% Current in A
try
    T_current.Max(1) = max(TT.Current(not(isnan(TT.Current))));
    T_current.Min(1) = min(TT.Current(not(isnan(TT.Current))));
    T_current.Mean(1) = mean(TT.Current(not(isnan(TT.Current))));
    T_current.("Standard deviation")(1) = std(TT.Current(not(isnan(TT.Current))));
    T_current.Skewness(1) = skewness(TT.Current(not(isnan(TT.Current))));
    T_current.Kurtosis(1) = kurtosis(TT.Current(not(isnan(TT.Current))));
end

try
    [current_N,current_edges] = histcounts(TT.Current(not(isnan(TT.Current))));
    current_edges = movmean(current_edges,2);
    current_edges = current_edges(2:end);
    current_N = current_N./sum(not(isnan(TT.Current(not(isnan(TT.Current))))));
    [current_pks,current_locs,current_w,current_p] = findpeaks(current_N);
    clear current_w
    [current_p,I] = sort(current_p,'descend');
    current_locs = current_locs(I);
    
    T_current.("Histogram 1. Max Value")(1) = current_edges(current_locs(1));
    T_current.("Histogram 1. Max Significance")(1) = current_N(current_locs(1));
    T_current.("Histogram 2. Max Value")(1) = current_edges(current_locs(2));
    T_current.("Histogram 2. Max Significance")(1) = current_N(current_locs(2));
    T_current.("Histogram 3. Max Value")(1) = current_edges(current_locs(3));
    T_current.("Histogram 3. Max Significance")(1) = current_N(current_locs(3));
    T_current.("Histogram 4. Max Value")(1) = current_edges(current_locs(4));
    T_current.("Histogram 4. Max Significance")(1) = current_N(current_locs(4));
end

% Temperature in °C
try
    T_temperature.Max(1) = max(TT.Temperature(not(isnan(TT.Temperature))));
    T_temperature.Min(1) = min(TT.Temperature(not(isnan(TT.Temperature))));
end

try
    [temp_N,temp_edges] = histcounts(TT.Temperature(not(isnan(TT.Temperature))));
    temp_edges = movmean(temp_edges,2);
    temp_edges = temp_edges(2:end);
    temp_N = temp_N./sum(not(isnan(TT.Temperature(not(isnan(TT.Temperature))))));
    [temp_pks,temp_locs,temp_w,temp_p] = findpeaks(temp_N);
    clear temp_w
    [temp_p,I] = sort(temp_p,'descend');
    temp_locs = temp_locs(I);
    
    T_temperature.("Histogram 1. Max Value")(1) = temp_edges(temp_locs(1));
    T_temperature.("Histogram 1. Max Significance")(1) = temp_N(temp_locs(1));
    T_temperature.("Histogram 2. Max Value")(1) = temp_edges(temp_locs(2));
    T_temperature.("Histogram 2. Max Significance")(1) = temp_N(temp_locs(2));
    T_temperature.("Histogram 3. Max Value")(1) = temp_edges(temp_locs(3));
    T_temperature.("Histogram 3. Max Significance")(1) = temp_N(temp_locs(3));
    T_temperature.("Histogram 4. Max Value")(1) = temp_edges(temp_locs(4));
    T_temperature.("Histogram 4. Max Significance")(1) = temp_N(temp_locs(4));
    T_temperature.("Histogram 5. Max Value")(1) = temp_edges(temp_locs(5));
    T_temperature.("Histogram 5. Max Significance")(1) = temp_N(temp_locs(5));
    T_temperature.("Histogram 6. Max Value")(1) = temp_edges(temp_locs(6));
    T_temperature.("Histogram 6. Max Significance")(1) = temp_N(temp_locs(6));
    T_temperature.("Histogram 7. Max Value")(1) = temp_edges(temp_locs(7));
    T_temperature.("Histogram 7. Max Significance")(1) = temp_N(temp_locs(7));
    T_temperature.("Histogram 8. Max Value")(1) = temp_edges(temp_locs(8));
    T_temperature.("Histogram 8. Max Significance")(1) = temp_N(temp_locs(8));
    T_temperature.("Histogram 9. Max Value")(1) = temp_edges(temp_locs(9));
    T_temperature.("Histogram 9. Max Significance")(1) = temp_N(temp_locs(9));
    T_temperature.("Histogram 10. Max Value")(1) = temp_edges(temp_locs(10));
    T_temperature.("Histogram 10. Max Significance")(1) = temp_N(temp_locs(10));
end




% Combine everything into a struct
T = struct('T_cell_name',T_cell_name, 'T_time',T_time, 'T_capacity',T_capacity,'T_ah_throughput',T_ah_throughput,'T_voltage',T_voltage,'T_current',T_current,'T_temperature',T_temperature);

%% finished, display the required time for the complete cell
fprintf('%s Extract_Statistics_function:\t\t\t %f s\n',cell_name,toc);
end