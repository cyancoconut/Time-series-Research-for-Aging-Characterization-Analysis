%% c_combine_timetables.m
% This is only necessary if your tests have used different names for the
% same cell. E.g. "Bat2020_G1_ISEA33_EIS" and "Bat2020_G1_ISEA33". NEVER
% do so!!! Only use one name in ahjo for the same cell.

function c_combine_timetables( path_to_data,parallel,compression, csv_export)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
        parallel (1,1) logical = 1;             % run parallel ?
        compression (1,1) logical = 1;          % compress the .mat files or 0 -> '-nocompression'?
        csv_export (1,1) logical = 0;           % export as csv ? (takes much longer!)
    end


% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"timeseries/"));


% load the existing "cells.xlsx" file in the timeseries folder
filename =strcat(path_to_data,"cells.xlsx");

% if it doesn't exist abort
try
    T = readtable(filename);
catch
    disp(strcat("No file found:", path_to_data, "cells.xlsx"))
    disp("Consider to run ""list_available_cells.m"" and afterward ""convert_to_timetable.m""")
    return
end

if parallel == 1
    T = T(randperm(size(T,1)), :);
end

file_equal_matrix = false(size(T,1),size(T,1));
names_deleted = [];
rows_deleted = [];

name_vector_with_project = string(T(:,1).Cell_Name);
name_vector = strings(size(name_vector_with_project));
for y = 1:size(name_vector,1)
    tmp_string = strsplit(name_vector_with_project(y,:),'-PROJECT-');
    name_vector(y,1) = tmp_string(1);
end


if parallel
    parfor y = 1:size(T,1)
        file_equal_matrix(y,:) = (contains(string(T(:,1).Cell_Name),name_vector(y,1),'IgnoreCase',true) & contains(name_vector_with_project,"_EIS",'IgnoreCase',true));% | (contains(name_vector,string(T(y,1).Cell_Name),'IgnoreCase',true) & contains(name_vector,"_gfu-she",'IgnoreCase',true));
    end
else
    for y = 1:size(T,1)
        file_equal_matrix(y,:) = (contains(string(T(:,1).Cell_Name),name_vector(y,1),'IgnoreCase',true) & contains(name_vector_with_project,"_EIS",'IgnoreCase',true));% | (contains(name_vector,string(T(y,1).Cell_Name),'IgnoreCase',true) & contains(name_vector,"_gfu-she",'IgnoreCase',true));
    end
end


for y = 1:size(T,1)
    file_equal_matrix(y,y) = true;
    if sum(file_equal_matrix(y,:)) > 1
        name_y = string(T(y,1).Cell_Name);
        name_x = string(T(file_equal_matrix(y,:).',1).Cell_Name);

        size_x = strlength(name_x);
        size_y = strlength(name_y);
        diff_size = size_x-size_y;
        diff_size = not(diff_size ==4 | diff_size == 0);
        
        name_x(diff_size) = [];

        if length(name_x) < 2
            continue
        end

        TT_combined = timetable();
        TT_combined.Time.TimeZone = 'Europe/Berlin';
        TT_combined.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        for x = 1:size(name_x,1)
            try
                load(strcat(path_to_data,'timeseries\',name_x(x),'.mat'),'TT');
                try
                    TT.EIS_Frequency(1);
                    TT = removevars(TT,'Duration');
                    try
                        TT = removevars(TT,{'Capcity','Capacity_Current'});
                    end
                end
                TT_combined = synchronize(TT_combined, TT);
                delete(strcat(path_to_data,'/timeseries/',name_x(x),'.mat'));
                names_deleted = [names_deleted; name_x(x)];
                rows_deleted = [rows_deleted; find(strcmp(string(T(:,1).Cell_Name),name_x(x)))];
            catch
                names_deleted = [names_deleted; name_x(x)];
                rows_deleted = [rows_deleted; find(strcmp(string(T(:,1).Cell_Name),name_x(x)))];
                continue;
            end
            file_equal_matrix(y,x) = false;
            file_equal_matrix(x,y) = false;
        end
        TT = TT_combined;
        clear TT_combined
        % sort the data
        TT = sortrows(TT);
        % remove duplicate rows
        TT = unique(TT);
        % remove data point with equal time
        uniqueTimes = unique(TT.Time);
        TT = retime(TT,uniqueTimes,'fillwithmissing');
        
        [name_min_length, name_index] = min(strlength(name_x));
        
        cell_name = name_x(name_index);
        
        rows_deleted(rows_deleted == find(strcmp(string(T(:,1).Cell_Name),cell_name))) = [];
        names_deleted(strcmp(names_deleted,cell_name)) = [];
 
        if compression
            save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3');
        else
            save(strcat(path_to_data,'/timeseries/',cell_name,'.mat'),'TT','-v7.3','-nocompression');
        end

        if csv_export
            writetimetable(TT,strcat(path_to_data,'/export/',cell_name,'.csv'));
        end
        % finished, display the required time for the complete cell
        fprintf('%s was merged in %s.\n',names_deleted,cell_name);
        names_deleted = [];
    end
end





% modify the "cells.xlsx" to the new status after a backup of the old
% version
if height(T) > 0

    writetable(T,strcat(path_to_data,"cells_backup.xlsx"));

    T(rows_deleted,:)=[];
    delete(strcat(path_to_data,"cells.xlsx"));
    T = sortrows(T);
    writetable(T,strcat(path_to_data,"cells.xlsx"));
end

end