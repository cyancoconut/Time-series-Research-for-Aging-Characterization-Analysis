function TT = read_diga_eis(filename)
%READ_DIGA Summary of this function goes here
%   Detailed explanation goes here

if contains(filename,"invalid",'IgnoreCase',true)
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
    return;
end

if not(contains(filename,"=EIS" + digitsPattern(5),'IgnoreCase',true) | contains(filename,"=INS" + digitsPattern(5),'IgnoreCase',true))
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
    return;
end

try
    load(filename,'diga');
%     matObj = matfile(filename);
%     diga = matObj.diga;
    diga.daten;
catch
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
    return;
end

spalten_namen = string(fieldnames(diga.daten));


time_names = ["Time" , "Zeit"];
time_index = [];
for i = 1:length(time_names)
    index_tmp = strcmp(time_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        time_index = index_tmp;
        break
    end
end


eis_z_imag_names = ["Zimg1"];
eis_z_imag_index = [];
for i = 1:length(eis_z_imag_names)
    index_tmp = strcmp(eis_z_imag_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        eis_z_imag_index = index_tmp;
        break
    end
end


eis_z_real_names = ["Zreal1"];
eis_z_real_index = [];
for i = 1:length(eis_z_real_names)
    index_tmp = strcmp(eis_z_real_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        eis_z_real_index = index_tmp;
        break
    end
end


if isempty(eis_z_imag_index)
    eis_z_abs_mohm_names = ["Betrag"];
    eis_z_abs_mohm_index = [];
    for i = 1:length(eis_z_abs_mohm_names)
        index_tmp = strcmp(eis_z_abs_mohm_names(i),spalten_namen);
        index_tmp = find(index_tmp,1);
        if ~isempty(index_tmp)
            eis_z_abs_mohm_index = index_tmp;
            break
        end
    end
end

if isempty(eis_z_imag_index)
    eis_z_abs_ohm_names = ["Amp"];
    eis_z_abs_ohm_index = [];
    for i = 1:length(eis_z_abs_ohm_names)
        index_tmp = strcmp(eis_z_abs_ohm_names(i),spalten_namen);
        index_tmp = find(index_tmp,1);
        if ~isempty(index_tmp)
            eis_z_abs_ohm_index = index_tmp;
            break
        end
    end
end

if isempty(eis_z_imag_index)
    eis_z_phase_names = ["Phase"];
    eis_z_phase_index = [];
    for i = 1:length(eis_z_phase_names)
        index_tmp = strcmp(eis_z_phase_names(i),spalten_namen);
        index_tmp = find(index_tmp,1);
        if ~isempty(index_tmp)
            eis_z_phase_index = index_tmp;
            break
        end
    end
end


eis_freq_names = ["ActFreq","Freq"];
eis_freq_index = [];
for i = 1:length(eis_freq_names)
    index_tmp = strcmp(eis_freq_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        eis_freq_index = index_tmp;
        break
    end
end

output_channel_names = ["activeChan"];
output_channel_index = [];
for i = 1:length(output_channel_names)
    index_tmp = strcmp(output_channel_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        output_channel_index = index_tmp;
        break
    end
end

if isempty(eis_freq_index) || sum(diga.daten.(spalten_namen{eis_freq_index})) == 0
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
else
    % EIS data
    try
        if not(length(unique(diga.daten.(spalten_namen{time_index}))) == length(diga.daten.(spalten_namen{time_index})))
            diga.daten.(spalten_namen{time_index}) = diga.daten.(spalten_namen{time_index}) + (cumsum(ones(1,length(diga.daten.(spalten_namen{time_index})))) )/1000;
        end
        if isempty(eis_z_imag_index)
            if isempty(eis_z_abs_ohm_index)
        	    EIS_Z_abs = diga.daten.(spalten_namen{eis_z_abs_mohm_index}).';
                EIS_Z_phase = diga.daten.(spalten_namen{eis_z_phase_index}).'/180*pi;
            else
                EIS_Z_abs = diga.daten.(spalten_namen{eis_z_abs_ohm_index}).'*1000;
                EIS_Z_phase = diga.daten.(spalten_namen{eis_z_phase_index}).';
            end
            
        else
            EIS_Z = diga.daten.(spalten_namen{eis_z_real_index}).' + 1i.* diga.daten.(spalten_namen{eis_z_imag_index}).';
            EIS_Z_abs = abs(EIS_Z);
            EIS_Z_phase = angle(EIS_Z);
        end
        EIS_Z_abs = EIS_Z_abs ./1000;

        if isempty(output_channel_index)
            TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).','ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{eis_freq_index}).',EIS_Z_abs,EIS_Z_phase,'VariableNames',{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase'});
        else
            TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).','ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{eis_freq_index}).',EIS_Z_abs,EIS_Z_phase,diga.daten.(spalten_namen{output_channel_index}).','VariableNames',{'EIS_Frequency','EIS_Z_abs','EIS_Z_phase','EIS_Output_Channel'});
        end
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
end

end

