% configure the path of the data
path_to_data = "./../../data/";

% add helper functions to path
addpath("./10_helper_functions");




% get all files in the folder "converted"
all_files = dir(strcat(path_to_data,"eis_data/"));
all_files = all_files([all_files.isdir] == 0);
all_files_date = vertcat(all_files.datenum);
all_files = {all_files.name}.';
all_files = string(all_files);
all_files = struct("file_name",all_files);
all_files.date = datetime(all_files_date,'ConvertFrom','datenum','Format','yyyy-MM-dd HH:mm:ss.SSSSSS');

clear  all_files_date


eis_cells = all_files.file_name;
eis_cells = replace(eis_cells,'_eis.mat','');



for i = 1:length(eis_cells)
    preevaluate_EIS('cell_name',eis_cells(i),'path_to_data',path_to_data)
end


for i = 1:length(eis_cells)
    preevaluate_EIS_time('cell_name',eis_cells(i),'path_to_data',path_to_data)
end