function [] = resistance_estimation_function(varargin)
tic
close all

cell_name = "";
path_to_data = "";


%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=""",varargin{i+1},""";"))
end


% load the timeseries
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT'); 
catch
    warning(strcat("Timeseries could not be load: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    TT.Current(1); % test if data is accessible
catch
    warning(strcat("Current not available, seems not to be ageing data: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    TT = removevars(TT,{'Pulse_Resistance','Pulse_Resistance_Area','Pulse_Resistance_Duration','Pulse_Resistance_Current'});
end

% Derivation of the current of pulse that is considered to be constant.
current_derivation = 0.01;

% Upper limit of pulse duration in s
time_s_threshold = 120;


time_diff = milliseconds(diff(TT.Time))/1000;
time_s = [0; cumsum(time_diff)];
clear time_diff

Pulse_Resistance = NaN(length(time_s),1);
Pulse_Resistance_Area = NaN(length(time_s),1);
Pulse_Resistance_Duration = NaN(length(time_s),1);
Pulse_Resistance_Current = NaN(length(time_s),1);

current = TT.Current;
current(isnan(current))=0;

current_sign = sign(current);
current_sign_diff = [0; diff(current_sign)];

time_s_diff_diff = [diff(time_s(not(current_sign_diff==0))); 0];

time_s_diff = NaN(length(time_s),1);
time_s_diff(not(current_sign_diff==0)) = time_s_diff_diff;
time_s_diff_index = find(not(isnan(time_s_diff)));

time_s_diff_index = time_s_diff_index -1;

for i = 1:length(time_s_diff_index)-1
    capacity_current_diff =std(rmoutliers(current(time_s_diff_index(i)+1:time_s_diff_index(i+1)),"percentiles",[5 95]));% max(current(time_s_diff_index(i:i+1)))-min(current(time_s_diff_index(i:i+1)));
    current_tmp_mean = mean(current(time_s_diff_index(i)+1:time_s_diff_index(i+1)));
    time_s_diff_tmp = time_s(time_s_diff_index(i+1)) - time_s(time_s_diff_index(i)+1);

    if time_s_diff_tmp > time_s_threshold || abs(capacity_current_diff) > abs(current_derivation*current_tmp_mean) || time_s_diff_index(i+1) - time_s_diff_index(i) < 5
        continue
    end

    current_tmp = TT.Current(time_s_diff_index(i):time_s_diff_index(i+1));
    voltage_tmp = TT.Voltage(time_s_diff_index(i):time_s_diff_index(i+1));
    time_offset = min(cumsum(milliseconds(diff(TT.Time(time_s_diff_index(i):time_s_diff_index(i+1))))/1000));
    time_s_tmp = [0; cumsum(milliseconds(diff(TT.Time(time_s_diff_index(i):time_s_diff_index(i+1))))/1000)-time_offset];
    current_tmp_cum = max(abs(cumtrapz(time_s_tmp,current_tmp )));
    voltage_tmp_cum_sub_avg = voltage_tmp - min(voltage_tmp);
    if current_tmp_mean < 0
        voltage_tmp_cum_sub_avg = voltage_tmp_cum_sub_avg*-1;
        voltage_tmp_cum_sub_avg = voltage_tmp_cum_sub_avg -min(voltage_tmp_cum_sub_avg);
    end
    voltage_tmp_cum = max(abs(cumtrapz(time_s_tmp,voltage_tmp_cum_sub_avg )));
    voltage_tmp_min = min(voltage_tmp);
    voltage_tmp_max = max(voltage_tmp);
    duration_tmp_s = max(time_s_tmp) - min(time_s_tmp);

    Pulse_Resistance(time_s_diff_index(i)) = (voltage_tmp_max - voltage_tmp_min)/(abs(current_tmp_mean));
    Pulse_Resistance_Area(time_s_diff_index(i)) = voltage_tmp_cum/current_tmp_cum;
    Pulse_Resistance_Duration(time_s_diff_index(i)) = duration_tmp_s;
    Pulse_Resistance_Current(time_s_diff_index(i)) = current_tmp_mean;
end


Capacity_table = table(Pulse_Resistance,Pulse_Resistance_Area,Pulse_Resistance_Duration,Pulse_Resistance_Current,'VariableNames',{'Pulse_Resistance','Pulse_Resistance_Area','Pulse_Resistance_Duration','Pulse_Resistance_Current'});
TT = [TT,Capacity_table];

if compression == '1'
    save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3');
else
    save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3','-nocompression');
end

if csv_export == '1'
    writetimetable(TT,strcat(path_to_data,'/export/',cell_name,'.csv'));
end

% finished, display the required time for the complete cell
fprintf('%s resistance_estimation_function:\t\t\t %f s\n',cell_name,toc);
end