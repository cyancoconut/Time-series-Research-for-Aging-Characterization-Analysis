%% d_preevaluate_timetables.m

function X_export_timetables( path_to_data)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
    end


% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"converted/"));
mkdir(strcat(path_to_data,"timeseries/"));
mkdir(strcat(path_to_data,"export/"));

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


% get all cells to be converted
cell_names = string(T.Cell_Name(T.x_export_timetables == 1));
cell_names_converted = zeros(length(cell_names),1);


% convert all those cells

for i = 1:length(cell_names_converted)
    disp(strcat("start: ",cell_names(i)));
	export_timetable(cell_names(i),path_to_data);
    cell_names_converted(i) = 1;
end

% modify the "cells.xlsx" to the new status after a backup of the old
% version

if height(T) > 0
    writetable(T,strcat(path_to_data,"cells_backup.xlsx"));
end

for i = 1:length(cell_names_converted)
    if cell_names_converted(i) == 1
        T.x_export_timetables(T.Cell_Name == cell_names(i) ) = 0;
    end
end
T = sortrows(T);
writetable(T,strcat(path_to_data,"cells.xlsx"));

end