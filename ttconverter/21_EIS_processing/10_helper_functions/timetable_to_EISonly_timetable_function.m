function TT = timetable_to_EISonly_timetable_function(cell_name,path_to_data, compression, csv_export)

% start tic to display the total time requried at the end
tic
disp(strcat("start: ",cell_name))
% load the timeseries and check if EIS exist
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.EIS_Frequency(1);
catch
    TT = timetable();
    fprintf('%s is empty timetable_to_EISonly_timetable:\t\t\t %f s\n',cell_name,toc);
    return;
end

if sum(not(isnan(TT.EIS_Frequency))) == 0
    TT = timetable();
    fprintf('%s is empty timetable_to_EISonly_timetable:\t\t\t %f s\n',cell_name,toc);
    return;
end

try
    TT = removevars(TT,{'EIS_measurement_id'});
end

try
    TT = removevars(TT,{'SOC'});
end

try
    TT = removevars(TT,{'SOH'});
end

try
    TT = removevars(TT,{'x__index_level_0__'});
end

try
    TT.Name_SE = double(TT.Name_SE);
end

try
    TT.Pulse_Name;
end

parquet_export = 1;

% detail level
detail_level = 0;
% 0:eis only        
% 1:basic           Voltage, Current, Temperature 
% 2:with AH         Voltage, Current, Temperature, Duration, Ah_Counter, Ah_throughput
% 3:with AH, WH     Voltage, Current, Temperature, Duration, Ah_Counter, Wh_Counter, Ah_throughput, Wh_throughput
% 4:with AH and Cap Voltage, Current, Temperature, Duration, Ah_Counter, Ah_throughput, Capacity, Capacity_current
% 5:AH, WH, Cap     Voltage, Current, Temperature, Duration, Ah_Counter, Wh_Counter, Ah_throughput, Wh_throughput, Capacity, Capacity_current
try
    TT.Capacity(1);
    TT.Capacity_current(1);
    try
        TT.Ah_throughput(1);
        TT.Wh_throughput(1);
        TT.Ah_Counter(1);
        TT.Wh_Counter(1);

        TT.Duration(1);
        TT.Current(1);
        TT.Voltage(1);

        detail_level = 5;
    catch case_below_5
        % rwthrow(case_below_5)
        TT.Ah_throughput(1);
        TT.Ah_Counter(1);

        TT.Duration(1);
        TT.Current(1);
        TT.Voltage(1);

        detail_level = 4;
        
    end
catch case_below_4
    % rwthrow(case_below_4)
    try
        TT.Ah_throughput(1);
        TT.Wh_throughput(1);
        TT.Ah_Counter(1);
        TT.Wh_Counter(1);
        
        TT.Duration(1);
        TT.Current(1);
        TT.Voltage(1);
        
        detail_level = 3;
    catch case_below_3
        % rwthrow(case_below_3)
        try
            TT.Ah_throughput(1);
            TT.Ah_Counter(1);
        
            TT.Duration(1);
            TT.Current(1);
            TT.Voltage(1);
            detail_level = 2;
        catch case_below_2
            % rwthrow(case_below_2)
            try
                TT.Current(1);
                TT.Voltage(1);
                
                detail_level = 1;
            catch case_below_1
                % rwthrow(case_below_1)
                TT.EIS_Frequency(1);
                detail_level = 0;
            end
        end
    end
end

% only case 4 and 5 makes sense, otherwise capacity was not calculated
% before
switch detail_level
    case 0
        variables_to_add = {};
    case 1
        variables_to_add = {'Voltage','Current'};
    case 2
        variables_to_add = {'Voltage','Current','Ah_Counter','Ah_throughput'};
    case 3
        variables_to_add = {'Voltage','Current','Ah_Counter','Ah_throughput','Wh_Counter','Wh_throughput'};
    case 4
        variables_to_add = {'Voltage','Current','Ah_Counter','Ah_throughput','Capacity','Capacity_current'};
    case 5
        variables_to_add = {'Voltage','Current','Ah_Counter','Ah_throughput','Wh_Counter','Wh_throughput','Capacity','Capacity_current'};
end

% check if temperature values are available.
try
    TT.Temperature(1);
    variables_to_add{end+1} = 'Temperature';
end


try
    TT.SOC_py(1);
    variables_to_add{end+1} = 'SOC_py';
end

try
    TT.Pulse_Name(1);
    variables_to_add{end+1} = 'Pulse_Name';
end


% check if force values are available.
try
    TT.DMS(1);
    variables_to_add{end+1} = 'DMS';
    variables_to_add{end+1} = 'Kraft';
    variables_to_add{end+1} = 'Messuhr';
    variables_to_add{end+1} = 'Feinzeiger';
end

% get all frequencies available

%if twice a frequency in a row occures, take the 2nd
eis_nn = TT.EIS_Frequency>0;
EIS_measurement_id = NaN(length(TT.EIS_Frequency),1);
EIS_measurement_id(eis_nn) = sign([1; diff(TT.EIS_Frequency(eis_nn))]);
A = EIS_measurement_id(eis_nn);
while any(A(:)==0)
  ii1=A==0;
  ii2=circshift(ii1,[-1 0]);
  A(ii1)=A(ii2);
end
EIS_measurement_id(eis_nn) = A;
if sum(EIS_measurement_id(eis_nn)) >= 0 
    % frequencies increasing
    copy = EIS_measurement_id(eis_nn);
    copy(copy ==1) = 0;
    EIS_measurement_id(eis_nn) = copy;
else
    % frequencies decreasing
    copy = EIS_measurement_id(eis_nn);
    copy(copy ==-1) = 0;
    EIS_measurement_id(eis_nn) = copy;
end
EIS_measurement_id(eis_nn) = abs(EIS_measurement_id(eis_nn));
EIS_measurement_id(eis_nn)=  [1;diff(EIS_measurement_id(eis_nn))];
copy = EIS_measurement_id(eis_nn);
copy(copy<1) = 0;
EIS_measurement_id(eis_nn) = copy;
EIS_measurement_id(isnan(EIS_measurement_id)) = 0;
EIS_measurement_id(:) = cumsum(EIS_measurement_id(:));


if size(TT,1) < 2
    TT = timetable();
    TT.Time.TimeZone = 'Europe/Berlin';
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
end


TT = addvars(TT,EIS_measurement_id,'NewVariableNames',{'EIS_measurement_id'});

TT = rmmissing(TT,'MinNumMissing',length(TT.Properties.VariableNames)-1);

TT(diff(TT.Time)==0,:) = [];

TT = fillmissing(TT,'previous','DataVariables',{'EIS_measurement_id'});

freq_pointer = find(TT.EIS_Frequency>0);

if detail_level> 1
    AHcounter = [0;diff(TT.Ah_throughput)].*sign(TT.Current);
    AHcounter(isnan(AHcounter)) = 0;
    AHcounter = cumsum(AHcounter);
    
    SOC = NaN(length(AHcounter),1);
    SOH = NaN(length(AHcounter),1);
    
    
    last_measurement_id = 0;
    % go through all freq_points and search for soc
    % therefore search for latest full charge
    for freq_p_ind = 1:length(freq_pointer)
        if last_measurement_id == TT.EIS_measurement_id(freq_pointer(freq_p_ind))
            continue
        end
        last_measurement_id = TT.EIS_measurement_id(freq_pointer(freq_p_ind));
        % get the last discharge capacity
        last_capa_ind = find(TT.Capacity_current(1:freq_pointer(freq_p_ind))<0,1,'last');
        % check when the charge afterwards finished
        begin_of_full_charge_ind = last_capa_ind+1 + find(TT.Current(last_capa_ind+1:freq_pointer(freq_p_ind))>0,1,'first');
        % search for the end of the charge process
        end_of_full_charge_ind = begin_of_full_charge_ind+1+ find(TT.Current(begin_of_full_charge_ind+1:freq_pointer(freq_p_ind))<0,1,'first')-2;
        % normally this should be equal to the begin of soc setpoints
        begin_of_set_soc_discharge_ind = end_of_full_charge_ind;
    
    
        capa = TT.Capacity(last_capa_ind);
        AH_counter_begin_of_set_soc_discharge = AHcounter(begin_of_set_soc_discharge_ind);
        AH_counter_act = AHcounter(freq_pointer(freq_p_ind));
        delta_AH = abs(AH_counter_act - AH_counter_begin_of_set_soc_discharge);
        SOC_tmp = (capa - delta_AH)/capa;

    %     display(freq_p_ind)
        if not(isempty(SOC_tmp))
%             display("Voltage: "+TT.Voltage(find(not(isnan(TT.Voltage(1:freq_pointer(freq_p_ind)))),1,'last'),:)+", SoC: "+SOC_tmp)
            SOC(freq_pointer(freq_p_ind)) = SOC_tmp;
        end
        if not(isempty(capa))
            SOH(freq_pointer(freq_p_ind)) = capa;
        end
    end

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
                c_nom = T.Capacity_in_Ah(i);
                break
            end
        end
    else
        c_nom = max(SOH);
    end
    
    SOH = SOH./c_nom;
    TT = addvars(TT,SOH,'NewVariableNames',{'SOH'});
    TT = addvars(TT,SOC,'NewVariableNames',{'SOC'});
    
    TT = fillmissing(TT,'previous','DataVariables',{'SOC','SOH'});
    
    try
        TT.Capacity(TT.Capacity_current> 0) = NaN;
    end
end







TT = fillmissing(TT,'previous','DataVariables',variables_to_add);

TT = TT(freq_pointer,:);

freq_diff = [1, diff(sum(TT{:,{'EIS_Frequency',  'EIS_Z_abs','EIS_Z_phase'}}.'))];

freq_pointer = find(freq_diff ~= 0);

TT = TT(freq_pointer,:);

try
    TT = removevars(TT,{'EIS_measurement_id'});
end

EIS_measurement_id = [1;  abs(sign(diff( sign(diff(TT.EIS_Frequency)))))];
EIS_measurement_id(EIS_measurement_id == 0) = NaN;
EIS_measurement_id = [0; diff(EIS_measurement_id)];
EIS_measurement_id(EIS_measurement_id ==0) = 1;
EIS_measurement_id(isnan(EIS_measurement_id)) = 0;
EIS_measurement_id = cumsum(EIS_measurement_id);
EIS_measurement_id = [EIS_measurement_id ; EIS_measurement_id(end)];


if size(TT,1) < 2
    TT = timetable();
    TT.Time.TimeZone = 'Europe/Berlin';
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
end


TT = addvars(TT,EIS_measurement_id,'NewVariableNames',{'EIS_measurement_id'});

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

TT_eis = TT;

if isempty(TT_eis)
    fprintf('%s is empty timetable_to_EISonly_timetable:\t\t\t %f s\n',cell_name,toc);
    return
end
    
if compression
    save(strcat(path_to_data,'/eis_data/',cell_name,'_eis.mat'),'TT_eis','-v7.3');
else
    save(strcat(path_to_data,'/eis_data/',cell_name,'_eis.mat'),'TT_eis','-v7.3','-nocompression');
end

if csv_export
    writetimetable(TT_eis,strcat(path_to_data,'/export/',cell_name,'_eis.csv'));
end

if parquet_export
    parquetwrite(strcat(path_to_data,'export\',cell_name,'_eis.parquet'),TT_eis,'VariableCompression','gzip');
end

% finished, display the required time for the complete cell
fprintf('%s timetable_to_EISonly_timetable:\t\t\t %f s\n',cell_name,toc);


end