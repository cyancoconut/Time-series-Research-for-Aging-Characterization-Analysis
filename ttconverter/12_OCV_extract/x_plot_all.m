%% x_plot_all.m


% configure the path of the data
path_to_data = "./../../data/";
% run parallel ?
parallel = 0;
% compress the .mat files or 0 -> '-nocompression'?
compression = 1;

hide_name = 0;




% create all necessary folders normally they should already be there
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
cell_names = string(T.Cell_Name(T.a_OCV_extract == 1));
cell_names_converted = zeros(length(cell_names),1);


% convert all those cells
if parallel == 1
    parfor i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        x_plot_ocv('cell_name',strcat('"',cell_names(i),'"'),'hide_name', num2str(hide_name*1));
        cell_names_converted(i) = 1;
    end
else
    for i = 1:length(cell_names_converted)
        disp(strcat("start: ",cell_names(i)));
        x_plot_ocv('cell_name',strcat('"',cell_names(i),'"'),'hide_name', num2str(hide_name*1));
        cell_names_converted(i) = 1;
    end
end

% modify the "cells.xlsx" to the new status after a backup of the old
% version
T = sortrows(T);
if height(T) > 0
    writetable(T,strcat(path_to_data,"cells_backup.xlsx"));
end

for i = 1:length(cell_names_converted)
    if cell_names_converted(i) == 1
        T.Extract_OCV(T.Cell_Name == cell_names(i) ) = 0;
    end
end

writetable(T,strcat(path_to_data,"cells.xlsx"));