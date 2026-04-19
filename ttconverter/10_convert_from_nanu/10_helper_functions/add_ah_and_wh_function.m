function TT = add_ah_and_wh_function(cell_name,path_to_data_timeseries,compression,csv_export);

% start tic to display the total time requried at the end
tic

try
    load(strcat(path_to_data_timeseries,'timeseries\',cell_name,'.mat'));
    TT.Current;
catch
    fprintf('%s not found add_ah_and_wh_function:\t\t\t %f s\n',cell_name,toc);
    return
end

if isempty(TT)
    fprintf('%s is empty add_ah_and_wh_function:\t\t\t %f s\n',cell_name,toc);
    return
end

try
    TT = removevars(TT,{'Wh_throughput'});
end

try
    TT = removevars(TT,{'Ah_throughput'});
end

try
    TT = removevars(TT,{'Power'});
end


% Adding  Ah_throughput and Wh_throughput
time_threshold_in_h = 31/60/60;


TT_Ah_Wh = TT;
TT_Ah_Wh(isnan(TT_Ah_Wh.Current),:) = [];

time_h = TT_Ah_Wh.Time;
time_h = milliseconds(diff(time_h))/1000/60/60;
time_h = [0; cumsum(time_h)];
time_h_diff = diff(time_h);
time_h_diff = [0;time_h_diff];
ind_interrupts = time_h_diff >time_threshold_in_h;
ind_interrupts = ind_interrupts | circshift(ind_interrupts,-1);

try
    Power = TT_Ah_Wh.Current.*TT_Ah_Wh.Voltage;
    Power(isnan(Power))=0;
    TT_power = timetable(TT_Ah_Wh.Time,Power,'VariableNames',{'Power'});
    TT_power.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT_power.Time.TimeZone = 'Europe/Berlin';
    clear Power
end


try
    AHthroughput = TT_Ah_Wh.Current;
    AHthroughput(ind_interrupts) = 0;
    Time = TT_Ah_Wh.Time;

    Time(isnan(AHthroughput))=[];
    AHthroughput(isnan(AHthroughput))=[];
    
    AHthroughput = cumtrapz(time_h, abs(AHthroughput));
    AHthroughput = AHthroughput- min(AHthroughput);
    TT_Ah_new = timetable(Time,AHthroughput,'VariableNames',{'Ah_throughput'});
    TT_Ah_new.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT_Ah_new.Time.TimeZone = 'Europe/Berlin';
    clear AHthroughput
end

try
    WHthroughput_c = TT_Ah_Wh.Current;
    WHthroughput_v = TT_Ah_Wh.Voltage;
    WHthroughput_c(ind_interrupts) = 0; % only the current!
    Time = TT_Ah_Wh.Time;

    Time(isnan(WHthroughput_c))=[];
    
    WHthroughput_v(isnan(WHthroughput_c))=[];
    WHthroughput_c(isnan(WHthroughput_c))=[];
    
    WHthroughput_v(isnan(WHthroughput_v)) = 0;

    WHthroughput = WHthroughput_c.*WHthroughput_v;
    WHthroughput = cumtrapz(time_h, abs(WHthroughput));
    WHthroughput = WHthroughput- min(WHthroughput);
    TT_Wh_new = timetable(Time,WHthroughput,'VariableNames',{'Wh_throughput'});
    TT_Wh_new.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT_Wh_new.Time.TimeZone = 'Europe/Berlin';
    clear WHthroughput WHthroughput_c WHthroughput_v
end


clear TT_Ah_Wh time_h time_h_diff ind_interrupts

if not(exist('TT_Ah_new')) | not(exist('TT_Wh_new'))
    fprintf('%s failed !!! add_ah_and_wh_function:\t\t\t %f s\n',cell_name,toc);
    return
end

TT = synchronize(TT,TT_Ah_new,TT_Wh_new,TT_power);

clear TT_Ah_new TT_Wh_new TT_power
    
if compression
    save(strcat(path_to_data_timeseries,'/timeseries/',cell_name,'.mat'),'TT','-v7.3');
else
    save(strcat(path_to_data_timeseries,'/timeseries/',cell_name,'.mat'),'TT','-v7.3','-nocompression');
end

if csv_export
    writetimetable(TT,strcat(path_to_data_timeseries,'/export/',cell_name,'.csv'));
end

% finished, display the required time for the complete cell
fprintf('%s add_ah_and_wh_function:\t\t\t %f s\n',cell_name,toc);
end

