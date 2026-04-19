%% a_list_available_cells.m
% This functions scans through all data downloaded and converted from ahjo
% with nanu (dig2mat) and creats a "cells.xlsx" which is located the in the
% "path_to_data/timeseries" folder. Within the "cells.xlsx" you can choose
% which cells should later be evaluated / combined.

function a_list_available_cells( path_to_data, force_reset)
    arguments
        path_to_data string =  "./../../data/";
        force_reset (1,1) logical = 0;
    end

% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"converted/"));
mkdir(strcat(path_to_data,"timeseries/"));

% get all files in the folder "converted"
all_files = dir(strcat(path_to_data,"converted/**"));
all_files = all_files([all_files.isdir] == 0);
all_files_date = vertcat(all_files.datenum);
all_files = {all_files.name}.';
all_files = string(all_files);
all_files = struct("file_name",all_files);
all_files.date = datetime(all_files_date,'ConvertFrom','datenum','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');
all_files.cell_name = strrep(string(zeros(length(all_files.file_name),1)),"0","");
all_files.project_name = strrep(string(zeros(length(all_files.file_name),1)),"0","");

% extract project and cell name
for i = 1:length(all_files.file_name)
    names = strsplit(all_files.file_name(i),"=");
    all_files.cell_name(i) = names(2);
    all_files.project_name(i) = names(1);
end

clear names i all_files_date

% load the existing "cells.xlsx" file in the timeseries folder

filename =strcat(path_to_data,"cells.xlsx");

% if it doesn't exist create a dummy T_old variable
try
    T_old = readtable(filename);
catch
    disp(strcat("new file:", path_to_data, "cells.xlsx"))
    T_old = cell2table(cell(0,12), "VariableNames", {'Cell_Name', 'b_convert_to_timetable','d_clean_timetables','e_add_ah_and_wh_throughput','f_preevaluate_timetables','x_export_timetables','a_capacity_estimation','b_resistance_estimation','a_OCV_extract','a_timetable_to_EISonly_timetable','a_Extract_Statistics','a_capacity_fade_analysis'});
end

if force_reset ==1
    disp(strcat("new file:", path_to_data, "cells.xlsx"))
    T_old = cell2table(cell(0,12), "VariableNames", {'Cell_Name', 'b_convert_to_timetable','d_clean_timetables','e_add_ah_and_wh_throughput','f_preevaluate_timetables','x_export_timetables','a_capacity_estimation','b_resistance_estimation','a_OCV_extract','a_timetable_to_EISonly_timetable','a_Extract_Statistics','a_capacity_fade_analysis'});
end

% merge old and new cells in the "cells.xlsx"

call_names = strcat(all_files.cell_name,"-PROJECT-",all_files.project_name);
cells = unique(call_names);
cells = sortrows(cells,1);
cells_old = T_old.Cell_Name;
[cells_new_count, cells_new] = groupcounts([cells; cells_old]);
cells_new = cells_new(cells_new_count == 1);

% in case no new cell exis, end here; if new cells exist, create backup of
% old excel file
if isempty(cells_new)
    disp("no new cells found")
    clear all_files cells cells_new cells_new_count cells_old filename
    return
else
    if height(T_old) > 0
        writetable(T_old,strcat(path_to_data,"cells_backup.xlsx"));
    end
end


% set for all new cells all columns in the excel file to 1
cells_new_convert = ones(length(cells_new),1);
T_new = table(cells_new,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,cells_new_convert,'VariableNames',{'Cell_Name', 'b_convert_to_timetable','d_clean_timetables','e_add_ah_and_wh_throughput','f_preevaluate_timetables','x_export_timetables','a_capacity_estimation','b_resistance_estimation','a_OCV_extract','a_timetable_to_EISonly_timetable','a_Extract_Statistics','a_capacity_fade_analysis'});

% add all aditional columns in the excel
if length(T_old.Properties.VariableNames) ~= length(T_new.Properties.VariableNames)
    [new_variables_count, new_variables] = groupcounts(string([T_old.Properties.VariableNames, T_new.Properties.VariableNames]).');
    new_variables = new_variables(new_variables_count==1);
    
    for i = 1:length(T_old.Properties.VariableNames)-length(T_new.Properties.VariableNames)
        T_new = addvars(T_new,ones(length(cells_new),1),'NewVariableNames',{char(new_variables(i))});
    end
end


% merge and save the new version of the cells.xlsx
T_merge = [T_old;T_new];
T_merge = sortrows(T_merge);

writetable(T_merge,filename);

disp(strcat("new cell added: ",cells_new))


clear all_files cells cells_new cells_new_convert cells_old filename i new_variables new_variables_count cells_new_count
end