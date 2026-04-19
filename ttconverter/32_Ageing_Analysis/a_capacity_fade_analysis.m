%% a_capacity_fade_analysis.m
% All cells with a 1 in the "path_to_data/timeseries/cells.xlsx" will be 
% evaluated here.

function a_capacity_fade_analysis( path_to_data,parallel,compression, csv_export, normalize_capa)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
        parallel (1,1) logical = 0;             % run parallel ?
        compression (1,1) logical = 1;          % compress the .mat files or 0 -> '-nocompression'?
        csv_export (1,1) logical = 1;           % export as csv ? (takes much longer!)
        normalize_capa (1,1) logical = 0;       % export as csv ? (takes much longer!)
    end


% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"timeseries/"));
mkdir(strcat(path_to_data,"export/"));
mkdir(strcat(path_to_data,"figures/"));

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
cell_names = string(T.Cell_Name(T.a_capacity_fade_analysis == 1));
cell_names_converted = zeros(length(cell_names),1);

if not(isempty(cell_names))
    if size(cell_names,1) > 1
        cell_names_ahjo = split(cell_names,"-PROJECT");
        cell_names_ahjo = cell_names_ahjo(:,1);
    else
        cell_names_ahjo = split(cell_names,"-PROJECT");
        cell_names_ahjo = cell_names_ahjo(1);
    end
end

T_cell_name  = table(strings(length(cell_names_ahjo),1), 'VariableNames', {'Cell Name'});
T_cell_name.("Cell Name") = cell_names_ahjo;
T_nom_cap = get_nom_cap(T_cell_name,strcat(path_to_data,"cap_values.xlsx"));
[~,ii] = ismember(T_cell_name.("Cell Name"),T_nom_cap.("Cell Name"));
T_nom_cap = T_nom_cap(ii,:);
T_nom_cap = array2table(T_nom_cap.("Nominal Capacity in Ah"),"VariableNames",{'Nominal Capacity in Ah'});

T_nom_cap = [T_cell_name,T_nom_cap];

% T_nom_cap = [];

clear T_cell_name

% convert all those cells
if parallel == 1
    parfor i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        capacity_fade_analysis_function(cell_names(i),path_to_data,compression,csv_export,T_nom_cap,normalize_capa);
        cell_names_converted(i) = 1;
    end
else
    for i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        capacity_fade_analysis_function(cell_names(i),path_to_data,compression,csv_export,T_nom_cap,normalize_capa);
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
        T.a_capacity_fade_analysis(T.Cell_Name == cell_names(i) ) = 0;
    end
end
T = sortrows(T);
writetable(T,strcat(path_to_data,"cells.xlsx"));

end