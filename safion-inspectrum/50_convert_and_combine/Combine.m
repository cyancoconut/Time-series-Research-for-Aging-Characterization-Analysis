path_to_data = "Z:\Forschung\bgA\J7045_Futavis_Modultests\Daten\2_Data_Futavis\Zellen\Zyklische Alterung\";

export_path = path_to_data;

pattern_of_files_to_combine_all = [...
    "CELL_NR1_", "CELL_NR2_", "CELL_NR3_", "CELL_NR4_", "CELL_NR5_", "CELL_NR6_"...
    , "CELL_NR8_", "CELL_NR8_", "CELL_NR9_", "CELL_NR10_", "CELL_NR11_", "CELL_NR12_"...
    , "CELL_NR13_", "CELL_NR14_", "CELL_NR15_", "CELL_NR16_"];

for pattern_of_files_to_combine_idx = 1:size(pattern_of_files_to_combine_all)
    pattern_of_files_to_combine = pattern_of_files_to_combine_all(pattern_of_files_to_combine_idx);
    display(pattern_of_files_to_combine);

    file_ending = '.irf';
    files = dir(strcat(path_to_data,'/**/*',file_ending));
    
    TT_eis = timetable();
    
    for file = 1:size(files,1)
        S = readstruct(strcat(files(file).folder,'/',files(file).name),'FileType','xml');
        if not(contains(files(file).name,pattern_of_files_to_combine))
            continue
        end
        measurement_time = strcat(S.measurementResults.impedanceSpectrum.dateAttribute, " ", string(S.measurementResults.impedanceSpectrum.timeAttribute));
        measurement_time = datetime(measurement_time,'InputFormat','yyyy/MM/dd HH:mm:ss','Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin');
        voltage = S.measurementResults.batteryVoltage.voltageAttribute;
        temperature = S.measurementResults.temperatureSensor.temperatureAttribute;
        
        measurement = struct2table(S.measurementResults.impedanceSpectrum.spectrumData);
        EIS_Frequency = measurement.freqAttribute;
        EIS_Z_abs = abs(measurement.zReAttribute + 1i*measurement.zImAttribute);
        EIS_Z_phase = angle(measurement.zReAttribute + 1i*measurement.zImAttribute);
        
        measurement_time = (milliseconds(1:length(EIS_Frequency))/1000).'+measurement_time;
        
        Current = ones(length(EIS_Frequency),1)*S.measurementResults.impedanceSpectrum.currentAverageAttribute;
        Voltage = ones(length(EIS_Frequency),1)*voltage;
        Temperature = ones(length(EIS_Frequency),1)*temperature;
        AH_throughput = zeros(length(EIS_Frequency),1);
        Wh_throughput = AH_throughput;
        EIS_measurement_id = ones(length(EIS_Frequency),1)*file;
        Capacity = Wh_throughput;
        Capacity_current = Wh_throughput;
        SOH = Wh_throughput;
        SOC = Wh_throughput;
        
        TT_eis = [TT_eis; timetable(datetime(measurement_time,'Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin'),Current,Voltage,Temperature,EIS_Frequency,EIS_Z_abs,EIS_Z_phase,AH_throughput, Wh_throughput,Capacity, Capacity_current, SOH, SOC,EIS_measurement_id,'VariableNames',{'Current','Voltage','Temperature','EIS_Frequency','EIS_Z_abs','EIS_Z_phase','Ah_throughput','Wh_throughput','Capacity','Capacity_current','SOH','SOC','EIS_measurement_id'})];
        
    end
    
    file_name = strcat(pattern_of_files_to_combine,'_irf');%datestr(now, 'yyyy-mm-dd_HH-MM-SS-FFF'));
    mkdir(strcat(export_path,'/eis_data/'));
    save(strcat(export_path,'/eis_data/',file_name,'_eis.mat'),'TT_eis','-v7.3');
    writetimetable(TT_eis,strcat(export_path,'/eis_data/',file_name,'_eis.csv'));
end