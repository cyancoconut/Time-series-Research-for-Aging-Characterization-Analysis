function zz_read_back_from_parquet(path_to_data, compression, cells)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
        compression (1,1) logical = 1;          % compress the .mat files or 0 -> '-nocompression'?
        cells (:,1) string =  ["";""]; % configure the path of the data
    end
    

    for cell = 1:length(cells)
        tic
        TT = parquetread(path_to_data+"export/"+cells(cell)+".parquet",'OutputType','timetable');
        try
            TT.Prozedur = categorical(TT.Prozedur);
        end
        try
            TT.Zustand = categorical(TT.Zustand);
        end
        try
            TT.Name_Ahjo = categorical(TT.Name_Ahjo);
        end
        try
            TT.Name_SE = categorical(TT.Name_SE);
        end
        try
            TT.Pulse_Name = categorical(TT.Pulse_Name);
        end
        try
            TT.Capacity_Name = categorical(TT.Capacity_Name);
        end
        TT.Time = datetime(TT.Time,'Format','yyyy-MM-dd HH:mm:ss.SSSSSSZ', ...
            'TimeZone','Europe/Berlin');
        if compression
            save(strcat(path_to_data,'/timeseries/',cells(cell),'.mat'),'TT','-v7.3');
        else
            save(strcat(path_to_data,'/timeseries/',cells(cell),'.mat'),'TT','-v7.3','-nocompression');
        end
        fprintf('%s zz_read_back_from_parquet:\t\t\t %f s\n',cells(cell),toc);
    end   
end