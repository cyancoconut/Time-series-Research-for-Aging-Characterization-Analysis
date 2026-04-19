function TT = OCV_extract_function(varargin)

% start tic to display the total time requried at the end
tic


%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=",varargin{i+1},";"))
end


% load the timeseries and check if EIS exist
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.Capacity(1);
    TT.Current(1);
catch
    TT = timetable();
    TT.Time.TimeZone = 'Europe/Berlin';
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    fprintf('%s no capacity values found:\t\t\t %f s\n',cell_name,toc);
    return;
end


try
    TT = removevars(TT,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'});
end

% detail level
detail_level = 1;
% 1:basic           Voltage, Current, Temperature 
% 2:with AH         Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput
% 3:with AH, WH     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput
% 4:with AH and Cap Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput, Capacity, Capacity_current
% 5:AH, WH, Cap     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput, Capacity, Capacity_current
try
    TT.Capacity(1);
    try
        TT.Wh_Counter(1);
        detail_level = 5;
    catch
        detail_level = 4;
    end
catch
    try
        TT.Wh_Counter(1);
        detail_level = 3;
    catch
            try
                TT.Ah_throughput(1);
                detail_level = 2;
            catch
                detail_level = 1;
            end
    end
end

% only case 4 and 5 makes sense, otherwise capacity was not calculated
% before
switch detail_level
    case 1
        variables_to_add = {'Temperature', 'OCV_measurement_id'};
    case 2
        variables_to_add = {'Temperature', 'Ah_throughput',  'OCV_measurement_id','Ah_Counter'};
    case 3
        variables_to_add = {'Temperature', 'Ah_throughput', 'Wh_throughput','OCV_measurement_id','Ah_Counter','Wh_Counter'};
    case 4
        variables_to_add = {'Temperature', 'Capacity','Ah_throughput', 'Capacity_current','OCV_measurement_id','Ah_Counter'};
    case 5
        variables_to_add = {'Temperature', 'Capacity','Ah_throughput', 'Wh_throughput', 'Capacity_current','OCV_measurement_id','Ah_Counter','Wh_Counter'};
end





% get all ocv points

ocv_pointer = find(TT.Capacity>0);


OCV_measurement_id = NaN(size(TT,1),1);


for ocv_index = 1:length(ocv_pointer)
    OCV_measurement_id(ocv_pointer(ocv_index)) = ocv_index;
end

ocv_m_id_table = table(OCV_measurement_id,'VariableNames',{'OCV_measurement_id'});
TT = [TT,ocv_m_id_table];

TT = fillmissing(TT,'previous','DataVariables',variables_to_add);

current_sign = sign(TT.Current);

copy_index = false(length(TT.Current),1);

% TT_new = timetable();
for ocv_index = 1:length(ocv_pointer)
    current_sign(1:ocv_pointer(ocv_index)) = NaN;
    
    % charge or discharge?
    if TT.Capacity_current(ocv_pointer(ocv_index)) > 0 % charge
        charge_start = ocv_pointer(ocv_index);

        % find where charge stops
        charge_stop = find(current_sign<1,1);
    
        % combine everything
%         TT_new = [TT_new; TT(charge_start:charge_stop,:)];

        copy_index(charge_start:charge_stop) = true;

    else % discharge
        discharge_start = ocv_pointer(ocv_index);
        
        % find where discharge stops
        discharge_stop = find(current_sign>-1,1);
    
        % combine everything
%         TT_new = [TT_new; TT(discharge_start:discharge_stop,:)];

        copy_index(discharge_start:discharge_stop) = true;
    end
end

TT_new = TT(copy_index,:);

TT = TT_new;
clear TT_new

if size(TT,1) < 2
    TT = timetable();
    TT.Time.TimeZone = 'Europe/Berlin';
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
end


% remove data with missing time
TF  = ismissing(TT.Time);
TT = TT(~TF,:);
% sort the data
TT = sortrows(TT);
% remove duplicate rows
TT = unique(TT);
% remove data point with equal time
uniqueTimes = unique(TT.Time);
TT = retime(TT,uniqueTimes,'fillwithmissing');
% remove all data without time
TF  = ismissing(TT.Time);
TT = TT(~TF,:);

TT_ocv = TT;

if isempty(TT_ocv)
    fprintf('%s is empty OCV_extract_function:\t\t\t %f s\n',cell_name,toc);
    return
end
    
if compression
    save(strcat(path_to_data,'/ocv_data/',cell_name,'_ocv.mat'),'TT_ocv','-v7.3');
else
    save(strcat(path_to_data,'/ocv_data/',cell_name,'_ocv.mat'),'TT_ocv','-v7.3','-nocompression');
end

if delete_time
    try
        TT_ocv = timetable2table(TT_ocv);
        TT_ocv = removevars(TT_ocv,{'Time'});
    end
end

if csv_export
    if delete_time
        writetable(TT_ocv,strcat(path_to_data,'/export/',cell_name,'_ocv.csv'));
    else
        writetimetable(TT_ocv,strcat(path_to_data,'/export/',cell_name,'_ocv.csv'));
    end
end

if parquet_export
    parquetwrite(strcat(path_to_data,'/export/',cell_name,'_ocv.parquet'),TT_ocv,'VariableCompression','gzip');
end

% finished, display the required time for the complete cell
fprintf('%s OCV_extract_function:\t\t\t %f s\n',cell_name,toc);


end