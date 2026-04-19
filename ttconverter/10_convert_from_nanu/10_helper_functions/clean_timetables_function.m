function TT = clean_timetables_function(cell_name,path_to_data, compression, csv_export)
tic

if contains(cell_name,"Everlast_35E_")
    nice = 1;
elseif contains(cell_name,"J3590_Toshiba_LTO_")
    nice = 1;
elseif contains(cell_name,"LBE_Samsung_35E_")
    nice = 1;
elseif contains(cell_name,"litec40_")
    nice = 1;
elseif contains(cell_name,"ep sanyo ")
    nice = 1;
else
    % finished, display the required time for the complete cell
    fprintf('%s clean_timetables_function:\t\t\t %f s\n',cell_name,toc);
    return
end


% load the EIS timeseries 
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.Voltage(1); % test if data is accessible
catch
    warning(strcat("Timeseries file could not be load: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    if contains(cell_name,"Everlast_35E_")
        TT.Current(abs(TT.Current-4.5)< 0.3 ) = 0;
        TT.Current(abs(TT.Current)> 10.5 ) = NaN;
        
        TT.Voltage(TT.Voltage> 4.4 ) = NaN;
        TT.Voltage(TT.Voltage< 2.4 ) = NaN;
        
        TT.Temperature(TT.Temperature< -50 ) = NaN;
        TT.Temperature(TT.Temperature > 70 ) = NaN;
    end
    
    if contains(cell_name,"J3590_Toshiba_LTO_")
        TT.Temperature(TT.Temperature< -50 ) = NaN;
        TT.Temperature(TT.Temperature > 150 ) = NaN;
    end
    
    if contains(cell_name,"LBE_Samsung_35E_")
        TT.Temperature(TT.Temperature< -50 ) = NaN;
        TT.Temperature(TT.Temperature > 70 ) = NaN;
    end
    
    if contains(cell_name,"litec40_")
        TT.Temperature(TT.Temperature < -50 ) = NaN;
        TT.Temperature(TT.Temperature > 60 ) = NaN;
        TT.Voltage(TT.Voltage< 2.5 ) = NaN;
    end
    
    
    if contains(cell_name,"ep sanyo ")
        TT.Temperature(TT.Temperature< -50) = NaN;
        TT.Temperature(TT.Temperature > 80 ) = NaN;
    end
end


if compression
    save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3');
else
    save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3','-nocompression');
end

if csv_export
    writetimetable(TT,strcat(path_to_data,'/export/',cell_name,'.csv'));
end

% finished, display the required time for the complete cell
fprintf('%s clean_timetables_function:\t\t\t %f s\n',cell_name,toc);


end

