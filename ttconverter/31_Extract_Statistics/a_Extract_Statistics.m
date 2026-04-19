%% a_Extract_Statistics.m
% All cells with a 1 in the "path_to_data/timeseries/cells.xlsx" will be 
% evaluated here.

function a_Extract_Statistics( path_to_data,path_to_ahjo_cap_data_table,parallel,compression, csv_export)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
        path_to_ahjo_cap_data_table string =  path_to_data+"2021-12-14 Ahjo specimen cap.xlsx" % configure the path of the data
        parallel (1,1) logical = 0;             % run parallel ?
        compression (1,1) logical = 1;          % compress the .mat files or 0 -> '-nocompression'?
        csv_export (1,1) logical = 1;           % export as csv ? (takes much longer!)
    end


% add helper functions to path
addpath("./10_helper_functions");

% load the existing "cells.xlsx" file in the timeseries folder
filename =strcat(path_to_data,"cells.xlsx");

% if it doesn't exist abort
try
    T = readtable(filename);
catch
    disp(strcat("No file found:", path_to_data, "cells.xlsx"))
    disp("Consider to run ""list_available_cells.m""")
    return
end

if parallel == 1
    T = T(randperm(size(T,1)), :);
end


% get all cells to be converted
cell_names = string(T.Cell_Name(T.a_Extract_Statistics == 1));
cell_names_converted = zeros(length(cell_names),1);



% _______________________________________________________________________
T_cell_name  = table(strings(length(cell_names),1), 'VariableNames', {'Cell Name'});
T_time  = table(nan(length(cell_names),1),nan(length(cell_names),1), 'VariableNames', {'Start Date', 'Duration in days'});
T_capacity  = table(nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis'});
T_ah_throughput  = table(nan(length(cell_names),1), 'VariableNames', {'Max'});
T_voltage  = table(nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis','Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance'});
T_current  = table(nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1), 'VariableNames', {'Max', 'Min', 'Mean', 'Standard deviation', 'Skewness','Kurtosis','Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance'});
T_temperature  = table(nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1),nan(length(cell_names),1), 'VariableNames', {'Max', 'Min', 'Histogram 1. Max Value','Histogram 1. Max Significance','Histogram 2. Max Value','Histogram 2. Max Significance','Histogram 3. Max Value','Histogram 3. Max Significance','Histogram 4. Max Value','Histogram 4. Max Significance','Histogram 5. Max Value','Histogram 5. Max Significance','Histogram 6. Max Value','Histogram 6. Max Significance','Histogram 7. Max Value','Histogram 7. Max Significance','Histogram 8. Max Value','Histogram 8. Max Significance','Histogram 9. Max Value','Histogram 9. Max Significance','Histogram 10. Max Value','Histogram 10. Max Significance'});
% _______________________________________________________________________


% convert all those cells
if parallel == 1
    parfor i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));

        T_stat = Extract_Statistics_function('cell_name',strcat('"',cell_names(i),'"'),'path_to_data',strcat('"',path_to_data,'"'));
             
        T_cell_name(i,:)  = T_stat(1).T_cell_name;
        T_time(i,:)  = T_stat(1).T_time;
        T_capacity(i,:)  = T_stat(1).T_capacity;
        T_ah_throughput(i,:)  = T_stat(1).T_ah_throughput;
        T_voltage(i,:)  = T_stat(1).T_voltage;
        T_current(i,:)  = T_stat(1).T_current;
        T_temperature(i,:)  = T_stat(1).T_temperature;
        
        cell_names_converted(i) = 1;
    end
else
    for i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));

        T_stat = Extract_Statistics_function('cell_name',strcat('"',cell_names(i),'"'),'path_to_data',strcat('"',path_to_data,'"'));
             
        T_cell_name(i,:)  = T_stat(1).T_cell_name;
        T_time(i,:)  = T_stat(1).T_time;
        T_capacity(i,:)  = T_stat(1).T_capacity;
        T_ah_throughput(i,:)  = T_stat(1).T_ah_throughput;
        T_voltage(i,:)  = T_stat(1).T_voltage;
        T_current(i,:)  = T_stat(1).T_current;
        T_temperature(i,:)  = T_stat(1).T_temperature;
        
        cell_names_converted(i) = 1;
    end
end

% modify the "cells.xlsx" to the new status after a backup of the old
% version
if height(T) > 0
    writetable(T,strcat(path_to_data,"cells_backup.xlsx"));
end

for i = 1:length(cell_names_converted)
    if cell_names_converted(i) == 1
        T.a_Extract_Statistics(T.Cell_Name == cell_names(i) ) = 0;
    end
end

T = sortrows(T);
writetable(T,strcat(path_to_data,"cells.xlsx"));

T_nom_cap = get_nom_cap(T_cell_name,path_to_ahjo_cap_data_table);
[~,ii] = ismember(T_cell_name.("Cell Name"),T_nom_cap.("Cell Name"));
T_nom_cap = T_nom_cap(ii,:);
T_nom_cap = array2table(T_nom_cap.("Nominal Capacity in Ah"),"VariableNames",{'Nominal Capacity in Ah'});

filename = "Cell_statistics";

T_cell_stat = table(T_cell_name,T_time,T_nom_cap,T_capacity,T_ah_throughput,T_voltage,T_current,T_temperature,'VariableNames',{'Cell Name','Time','Nominal Capacity in Ah','Capacity in Ah','Ah Throughput in Ah','Voltage in V','Current in A','Temperature in °C'});

if compression
    save(strcat(path_to_data,'/',filename','.mat'),'T_cell_name','T_time','T_nom_cap','T_capacity','T_ah_throughput','T_voltage','T_current','T_temperature','-v7.3');
else
    save(strcat(path_to_data,'/',filename','.mat'),'T_cell_name','T_time','T_nom_cap','T_capacity','T_ah_throughput','T_voltage','T_current','T_temperature','-v7.3','-nocompression');
end

T_header = array2table(strings(1,size(splitvars(T_cell_stat),2)));
T_header(1,"Var1") = table("Cell Name");
T_header(1,"Var2") = table("Time");
T_header(1,"Var4") = table("Nominal Capacity in Ah");
T_header(1,"Var5") = table("Capacity in Ah");
T_header(1,"Var11") = table("Ah Throughput in Ah");
T_header(1,"Var12") = table("Voltage in V");
T_header(1,"Var26") = table("Current in A");
T_header(1,"Var40") = table("Temperature in °C");
writetable(splitvars(T_header),strcat(path_to_data,'/',filename','.xlsx'),"Range",'A1','WriteVariableNames',false);

writetable(splitvars(T_cell_name),strcat(path_to_data,'/',filename','.xlsx'),"Range",'A2');
writetable(splitvars(T_time),strcat(path_to_data,'/',filename','.xlsx'),"Range",'B2');
writetable(splitvars(T_nom_cap),strcat(path_to_data,'/',filename','.xlsx'),"Range",'D2');
writetable(splitvars(T_capacity),strcat(path_to_data,'/',filename','.xlsx'),"Range",'E2');
writetable(splitvars(T_ah_throughput),strcat(path_to_data,'/',filename','.xlsx'),"Range",'K2');
writetable(splitvars(T_voltage),strcat(path_to_data,'/',filename','.xlsx'),"Range",'L2');
writetable(splitvars(T_current),strcat(path_to_data,'/',filename','.xlsx'),"Range",'Z2');
writetable(splitvars(T_temperature),strcat(path_to_data,'/',filename','.xlsx'),"Range",'AN2');


if csv_export
    mkdir(strcat(path_to_data,"export/"));
    writetable(splitvars(T_cell_stat),strcat(path_to_data,'/export/',filename','.csv'));
end

end