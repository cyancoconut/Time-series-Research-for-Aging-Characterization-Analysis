%% b_convert_to_timetable.m
% All cells with a 1 in the "path_to_data/timeseries/cells.xlsx" will be 
% combined here. It will create a timetable containing time, voltage, 
% current, temperature and if available EIS measurement results.

function b_convert_to_timetable( path_to_data_converted, path_to_data_timeseries, parallel,compression, csv_export, eis,dms)
    arguments
        path_to_data_converted string =  "./../../data/";
        path_to_data_timeseries string =  "./../../data/";
        parallel (1,1) logical = 1;
        compression (1,1) logical = 1;
        csv_export (1,1) logical = 0;
        eis (1,1) logical = 1;
        dms (1,1) logical = 0;
    end


% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data_converted,"converted/"));
mkdir(strcat(path_to_data_timeseries,"timeseries/"));
mkdir(strcat(path_to_data_timeseries,"export/"));

% load the existing "cells.xlsx" file in the timeseries folder
filename =strcat(path_to_data_converted,"cells.xlsx");

% if it doesn't exist abort
try
    T = readtable(filename);
catch
    disp(strcat("No file found:", path_to_data_converted, "cells.xlsx"))
    disp("Consider to run ""list_available_cells.m""")
    return
end

if parallel == 1
    T = T(randperm(size(T,1)), :);
end





% get all cells to be converted
cell_names = string(T.Cell_Name(T.b_convert_to_timetable == 1));
cell_names_converted = zeros(length(cell_names),1);


% convert all those cells
if parallel == 1
    parfor i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        convert_to_timetable_function(cell_names(i),eis,dms,path_to_data_converted,path_to_data_timeseries,compression,csv_export);
        cell_names_converted(i) = 1;
    end
else
    for i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        convert_to_timetable_function(cell_names(i),eis,dms,path_to_data_converted,path_to_data_timeseries,compression,csv_export);
        cell_names_converted(i) = 1;
    end
end

% modify the "cells.xlsx" to the new status after a backup of the old
% version

if height(T) > 0
    writetable(T,strcat(path_to_data_converted,"cells_backup.xlsx"));
end

for i = 1:length(cell_names_converted)
    if cell_names_converted(i) == 1
        T.b_convert_to_timetable(T.Cell_Name == cell_names(i) ) = 0;
    end
end
T = sortrows(T);
writetable(T,strcat(path_to_data_converted,"cells.xlsx"));

end