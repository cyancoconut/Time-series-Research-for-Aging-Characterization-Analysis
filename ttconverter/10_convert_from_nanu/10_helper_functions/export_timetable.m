function export_timetable(cell_name,path_to_data)

tic

force = 1;

try
    load(strcat(path_to_data,'export\',cell_name,'.csv'))
    if ~force
        return
    end
end

try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'))
catch
    return
end


% TT_export = timetable(TT.Time,TT.Voltage,TT.Current,TT.Temperature,'VariableNames',{'Voltage','Current','Temperature'});

TT_export = TT;

TF  = ismissing(TT_export.Time);
TT_export = TT_export(~TF,:);

% sort the data
TT_export = sortrows(TT_export);

% remove duplicate rows
TT_export = unique(TT_export);

TT = TT_export;

TT = rmmissing(TT,1,'MinNumMissing',length(TT.Properties.VariableNames));


% writetimetable(TT,strcat(path_to_data,'export\',cell_name,'.csv'));
parquetwrite(strcat(path_to_data,'export\',cell_name,'.parquet'),TT,'VariableCompression','gzip');

% save(strcat('./../../data\export\',cell_name,'.mat'),'TT','-v7.3');
% save(strcat('./../../data\export\',cell_name,'_nocompression.mat'),'TT','-v7.3','-nocompression');
% 
% h5create(strcat('./../../data\export\EVERLASTING.h5'), strcat('/',cell_name,'/time'), length(TT.Time))
% h5create(strcat('./../../data\export\EVERLASTING.h5'), strcat('/',cell_name,'/voltage'), length(TT.Voltage))
% h5create(strcat('./../../data\export\EVERLASTING.h5'), strcat('/',cell_name,'/current'), length(TT.Current))
% h5create(strcat('./../../data\export\EVERLASTING.h5'), strcat('/',cell_name,'/temperature'), length(TT.Temperature))
% 
% h5write(strcat('./../../data\export\EVERLASTING.h5'),strcat('/',cell_name,'/time'),datenum(TT.Time))
% h5write(strcat('./../../data\export\EVERLASTING.h5'),strcat('/',cell_name,'/voltage'),TT.Voltage)
% h5write(strcat('./../../data\export\EVERLASTING.h5'),strcat('/',cell_name,'/current'),TT.Current)
% h5write(strcat('./../../data\export\EVERLASTING.h5'),strcat('/',cell_name,'/temperature'),TT.Temperature)
% 
% h5create(strcat('./../../data\export\',cell_name,'.h5'), strcat('/time'), length(TT.Time))
% h5create(strcat('./../../data\export\',cell_name,'.h5'), strcat('/voltage'), length(TT.Voltage))
% h5create(strcat('./../../data\export\',cell_name,'.h5'), strcat('/current'), length(TT.Current))
% h5create(strcat('./../../data\export\',cell_name,'.h5'), strcat('/temperature'), length(TT.Temperature))
% 
% h5write(strcat('./../../data\export\',cell_name,'.h5'),strcat('/time'),datenum(TT.Time))
% h5write(strcat('./../../data\export\',cell_name,'.h5'),strcat('/voltage'),TT.Voltage)
% h5write(strcat('./../../data\export\',cell_name,'.h5'),strcat('/current'),TT.Current)
% h5write(strcat('./../../data\export\',cell_name,'.h5'),strcat('/temperature'),TT.Temperature)


fprintf('%s export_timetable finished after\t\t\t %f s\n',cell_name,toc);
end

