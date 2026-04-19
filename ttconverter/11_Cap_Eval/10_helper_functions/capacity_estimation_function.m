function [] = capacity_estimation_function(varargin)
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
    TT.Ah_throughput(1); % test if data is accessible
catch
    warning(strcat("Ah_throughput not available, seems not to be ageing data: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    TT = removevars(TT,{'Capacity','Capacity_current'});
end

NumberOfValues = sum(not(isnan(TT.Voltage)));

% percentage of the values, that are considered for the maxima
PercentageOfValues = 0.001;

% threshold, that the previous calculated maxima can be missed.
Threshold = 0.01;

time_h_threshold = 0.01;

if mean(TT.Voltage) < 5.0
    TT.Voltage(TT.Voltage > 5.0) = NaN;
end
if ceil(NumberOfValues*PercentageOfValues) >1
    Voltage_max = min(maxk(TT.Voltage,ceil(NumberOfValues*PercentageOfValues)));
else
    Voltage_max = max(TT.Voltage);
end
Voltage_max_limit = Voltage_max* (1-Threshold);
TT.Voltage(TT.Voltage < 0.1) = NaN;
if ceil(NumberOfValues*PercentageOfValues) >1
    Voltage_min = max(mink(TT.Voltage,ceil(NumberOfValues*PercentageOfValues)));
else
    Voltage_min = min(TT.Voltage);
end
Voltage_min_limit = Voltage_min* (1 + Threshold);

time_diff = milliseconds(diff(TT.Time))/1000/60/60;
time_h = [0; cumsum(time_diff)];
clear time_diff

capacity = NaN(length(time_h),1);
capacity_current = NaN(length(time_h),1);

current = TT.Current;
current(isnan(current))=0;
%ah = cumtrapz(time_h,current );
ah = TT.Ah_throughput;

current_sign = sign(current);
current_sign_diff = [0; diff(current_sign)];

time_h_diff_diff = [diff(time_h(not(current_sign_diff==0))); 0];

time_h_diff = NaN(length(time_h),1);
time_h_diff(not(current_sign_diff==0)) = time_h_diff_diff;
time_h_diff_index = find(not(isnan(time_h_diff)));

time_h_diff_index = time_h_diff_index -1;

% load the existing "cells.xlsx" file in the timeseries folder
filename =strcat(path_to_data,"cells_values.xlsx");

% if it doesn't exist abort
try
    T = readtable(filename);
catch
    disp(strcat("No file found:", path_to_data, "cells_values.xlsx"))
    disp("just add an empty file if you don't need to fix some thresholds")
    return
end

if contains(cell_name,T.Cell_Names(:))
    for i=1:length(T.Cell_Names)
        if contains(cell_name,T.Cell_Names(i))
            Voltage_max_limit = T.Voltage_max_limit(i);
            Voltage_min_limit = T.Voltage_min_limit(i);
            time_h_threshold = T.time_h_threshold(i);
            break
        end
    end
end



for i = 1:length(time_h_diff_index)-1
    capacity_current_tmp = mean(current(time_h_diff_index(i):time_h_diff_index(i+1)));
    if (length(current(time_h_diff_index(i):time_h_diff_index(i+1))) < 5) || (abs(mean(current(time_h_diff_index(i):time_h_diff_index(i+1))))<0.001)
        continue;
    end

    time_h_diff_tmp = time_h(time_h_diff_index(i+1)) - time_h(time_h_diff_index(i));

    
    % discharge (current <0)
    if capacity_current_tmp < 0 && time_h_diff_tmp > time_h_threshold && TT.Voltage(time_h_diff_index(i)) > Voltage_max_limit && TT.Voltage(time_h_diff_index(i+1)) <Voltage_min_limit && sum(current(time_h_diff_index(i):time_h_diff_index(i+1))>0) < ceil(length(current(time_h_diff_index(i):time_h_diff_index(i+1)))*0.01)
        capacity_current(time_h_diff_index(i)) = median(current(time_h_diff_index(i):time_h_diff_index(i+1)));
        capacity(time_h_diff_index(i)) = abs(ah(time_h_diff_index(i+1)) - ah(time_h_diff_index(i)));
    end
    % charge  (current >0)
    if capacity_current_tmp > 0 && time_h_diff_tmp > time_h_threshold && TT.Voltage(time_h_diff_index(i)) < Voltage_min_limit && TT.Voltage(time_h_diff_index(i+1)) > Voltage_max_limit && sum(current(time_h_diff_index(i):time_h_diff_index(i+1))<0) < ceil(length(current(time_h_diff_index(i):time_h_diff_index(i+1)))*0.01)
        capacity_current(time_h_diff_index(i)) = median(current(time_h_diff_index(i):time_h_diff_index(i+1)));
        capacity(time_h_diff_index(i)) = abs(ah(time_h_diff_index(i+1)) - ah(time_h_diff_index(i)));
    end
end


Capacity_table = table(capacity,capacity_current,'VariableNames',{'Capacity','Capacity_current'});
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
fprintf('%s capacity_estimation_function:\t\t\t %f s\n',cell_name,toc);
end