function TT = convert_to_timetable_function(cell_name,eis,dms,path_to_data_converted,path_to_data_timeseries,compression,csv_export);

% start tic to display the total time requried at the end
tic
cell_name_without_project = cell_name;
cell_name_without_project = strsplit(cell_name_without_project,'-PROJECT-');
cell_name_without_project = cell_name_without_project(1);


ds = fileDatastore(strcat(path_to_data_converted,'/converted','/**/*=',cell_name_without_project,'=*.mat'),'ReadFcn',@read_diga_time,'UniformRead',true );
ds = shuffle(ds);
TT_time = timetable();
TT_time.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
TT_time.Time.TimeZone = 'Europe/Berlin';


while hasdata(ds)
    try
        TT_tmp = ds.read();
        
        TT_time_colmissing = setdiff(TT_tmp.Properties.VariableNames, TT_time.Properties.VariableNames);
        TT_tmp_colmissing = setdiff(TT_time.Properties.VariableNames, TT_tmp.Properties.VariableNames);
        if not(isempty(TT_time_colmissing)) || not(isempty(TT_tmp_colmissing))
            for time_missing_index = 1:length(TT_time_colmissing)
                if (strcmp(TT_time_colmissing{time_missing_index},'Prozedur') || strcmp(TT_time_colmissing{time_missing_index},'Zustand'))
                    TT_time = [TT_time cell2table(cell(height(TT_time),1), 'VariableNames', {TT_time_colmissing{time_missing_index}})];
                else
                    TT_time = [TT_time array2table(nan(height(TT_time),1), 'VariableNames', {TT_time_colmissing{time_missing_index}})];
                end
            end
            for tmp_missing_index = 1:length(TT_tmp_colmissing)
                if (strcmp(TT_tmp_colmissing{tmp_missing_index},'Prozedur') || strcmp(TT_tmp_colmissing{tmp_missing_index},'Zustand'))
                    TT_tmp = [TT_tmp cell2table(cell(height(TT_tmp),1), 'VariableNames', {TT_tmp_colmissing{tmp_missing_index}})];
                else
                    TT_tmp = [TT_tmp array2table(nan(height(TT_tmp),1), 'VariableNames', {TT_tmp_colmissing{tmp_missing_index}})];
                end
            end
        end

        TT_time = [TT_time;TT_tmp];
    catch
        fprintf('%s TT_time = [TT_time;ds.read()]; failed\n',cell_name);
    end
end

ds = fileDatastore(strcat(path_to_data_converted,'/converted','/**/*=',cell_name_without_project,'=*.mat'),'ReadFcn',@read_diga_eis,'UniformRead',true );
ds = shuffle(ds);
TT_eis = timetable();
TT_eis.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
TT_eis.Time.TimeZone = 'Europe/Berlin';
if eis ==1
    while hasdata(ds)
        try
            TT_eis = [TT_eis;ds.read()];
        catch
            fprintf('%s TT_eis = [TT_eis;ds.read()];  failed\n',cell_name);
        end
    end
end


ds = fileDatastore(strcat(path_to_data_converted,'/converted','/**/*=',cell_name_without_project,'=*.mat'),'ReadFcn',@read_diga_dms,'UniformRead',true );
ds = shuffle(ds);
TT_dms = timetable();
TT_dms.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
TT_dms.Time.TimeZone = 'Europe/Berlin';
if dms ==1
    while hasdata(ds)
        try
            TT_dms = [TT_dms;ds.read()];
        catch
            fprintf('%s TT_dms = [TT_dms;ds.read()];  failed\n',cell_name);
        end
    end
end


% clean up
% combine
if isempty(TT_eis)
    if isempty(TT_dms)
        if isempty(TT_time)
            fprintf('%s is empty convert_to_timetable_function:\t\t\t %f s\n',cell_name,toc);
            return
        else
            TT_time = convertvars(TT_time,{'Current','Voltage'},'double');
            if any("Temperature" == string(TT_time.Properties.VariableNames))
                TT_time = convertvars(TT_time,{'Temperature'},'double');
            end
            TT = TT_time;
        end
    else
        if isempty(TT_time)
            fprintf('%s is empty convert_to_timetable_function:\t\t\t %f s\n',cell_name,toc);
            return
        else
            TT_time = convertvars(TT_time,{'Current','Voltage'},'double');
            if any("Temperature" == string(TT_time.Properties.VariableNames))
                TT_time = convertvars(TT_time,{'Temperature'},'double');
            end
            TT = synchronize(TT_time, TT_dms);
        end
    end
else
    if isempty(TT_dms)
        if isempty(TT_time)
            TT_eis = convertvars(TT_eis,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'},'double');
            TT = TT_eis;
        else
            TT_time = convertvars(TT_time,{'Current','Voltage'},'double');
            if any("Temperature" == string(TT_time.Properties.VariableNames))
                TT_time = convertvars(TT_time,{'Temperature'},'double');
            end
            TT_eis = convertvars(TT_eis,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'},'double');
            TT = synchronize(TT_time, TT_eis);
        end
    else
        if isempty(TT_time)
            TT_eis = convertvars(TT_eis,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'},'double');
            TT = synchronize(TT_eis, TT_dms);
        else
            TT_time = convertvars(TT_time,{'Current','Voltage'},'double');
            if any("Temperature" == string(TT_time.Properties.VariableNames))
                TT_time = convertvars(TT_time,{'Temperature'},'double');
            end
            TT_eis = convertvars(TT_eis,{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'},'double');
            TT = synchronize(TT_time, TT_eis,TT_dms);
        end
    end
end

clear TT_eis TT_time TT_dms

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

if isempty(TT)
    fprintf('%s is empty convert_to_timetable_function:\t\t\t %f s\n',cell_name,toc);
    return
end

TT.Duration = days(TT.Time(:)-TT.Time(1));

    
if compression
    save(strcat(path_to_data_timeseries,'/timeseries/',cell_name,'.mat'),'TT','-v7.3');
else
    save(strcat(path_to_data_timeseries,'/timeseries/',cell_name,'.mat'),'TT','-v7.3','-nocompression');
end

if csv_export
    writetimetable(TT,strcat(path_to_data_timeseries,'/export/',cell_name,'.csv'));
end

% finished, display the required time for the complete cell
fprintf('%s convert_to_timetable_function:\t\t\t %f s\n',cell_name,toc);
end

