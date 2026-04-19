% configure the path of the data
path_to_data = "F:\Backup_2023_02_20/";

mkdir(strcat(path_to_data,'export'));

% load the existing "Cell_Name_Lookup.xlsx" file in the data folder
filename =strcat(path_to_data,"Cell_Name_Lookup.xlsx");

% if it doesn't exist abort
try
    T = readtable(filename);
catch
    disp(strcat("No file found:", path_to_data, "Cell_Name_Lookup.xlsx"))
    return
end

% all_files = dir(strcat(path_to_data,"/export/"));
% all_files = all_files([all_files.isdir] == 0);
% all_files = {all_files.name}.';
% all_files = string(all_files);

% %% all "_ocv.parquet" in "export"
% for i = 1:height(T)
% %     file_index = contains(all_files,string(T{i,"CellNameISEA"}));
% 
%     source = strcat(path_to_data,"export/",string(T{i,"CellNameISEA"}),"_ocv.parquet");
%     destination = strcat(path_to_data,"export/",string(T{i,"CellNameExtern"}),"_ocv.parquet");
%     copyfile(source, destination)
% end

% %% all "_time_stat.png", "_pre-evaluation_time.png" and "_ocv.png" in "figures"
% for i = 1:height(T)
% %     file_index = contains(all_files,string(T{i,"CellNameISEA"}));
% 
%     source = strcat(path_to_data,"figures/",replace(string(T{i,"CellNameISEA"})," ","_"),"_time_stat.png");
%     destination = strcat(path_to_data,"figures/",string(T{i,"CellNameExtern"}),"_time_stat.png");
%     copyfile(source, destination)
% 
%     source = strcat(path_to_data,"figures/",replace(string(T{i,"CellNameISEA"})," ","_"),"_pre-evaluation_time.png");
%     destination = strcat(path_to_data,"figures/",string(T{i,"CellNameExtern"}),"_pre-evaluation_time.png");
%     copyfile(source, destination)
% 
%     source = strcat(path_to_data,"figures/",replace(string(T{i,"CellNameISEA"})," ","_"),"_ocv.png");
%     destination = strcat(path_to_data,"figures/",string(T{i,"CellNameExtern"}),"_ocv.png");
%     copyfile(source, destination)
% end

for i = 1:height(T)
    source = strcat(path_to_data,"timeseries/",replace(string(T{i,"CellNameISEA"}),"",""),".mat");
    destination = strcat(path_to_data,"export/",string(T{i,"CellNameExtern"}),".mat");
%     copyfile(source, destination)
    load(source);
    destination_csv = replace(destination,'.mat','.csv');
%     writetimetable(TT,destination_csv);
    destination_parquet = replace(destination,'.mat','.parquet');
    parquetwrite(destination_parquet,TT,'VariableCompression','gzip');
end
