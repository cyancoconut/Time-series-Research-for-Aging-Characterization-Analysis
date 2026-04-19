function TT = read_diga_time(filename)
%READ_DIGA Summary of this function goes here
%   Detailed explanation goes here

if contains(filename,"invalid",'IgnoreCase',true)
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
    disp("removed!")
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

% if isempty(diga.spalten_namen)
%     TT = timetable();
%     TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
%     TT.Time.TimeZone = 'Europe/Berlin';
%     return;
% end

spalten_namen = string(fieldnames(diga.daten));


% check if eis file, then kick

eis_freq_names = "ActFreq";
eis_freq_index = [];
for i = 1:length(eis_freq_names)
    index_tmp = strcmp(eis_freq_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        eis_freq_index = index_tmp;
        break
    end
end

if not(isempty(eis_freq_index))
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
    return;
end

schritt_names = ["Schritt"];
schritt_index = [];
for i = 1:length(schritt_names)
    index_tmp = strcmp(schritt_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        schritt_index = index_tmp;
        break
    end
end

zustand_names = ["Zustand"];
zustand_index = [];
for i = 1:length(zustand_names)
    index_tmp = strcmp(zustand_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        zustand_index = index_tmp;
        break
    end
end

schrittdauer_names = ["Schrittdauer"];
schrittdauer_index = [];
for i = 1:length(schrittdauer_names)
    index_tmp = strcmp(schrittdauer_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        schrittdauer_index = index_tmp;
        break
    end
end

zyklus_names = ["Zyklus"];
zyklus_index = [];
for i = 1:length(zyklus_names)
    index_tmp = strcmp(zyklus_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        zyklus_index = index_tmp;
        break
    end
end

zyklusebene_names = ["Zyklusebene"];
zyklusebene_index = [];
for i = 1:length(zyklusebene_names)
    index_tmp = strcmp(zyklusebene_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        zyklusebene_index = index_tmp;
        break
    end
end

prozedurebene_names = ["Prozedurebene"];
prozedurebene_index = [];
for i = 1:length(prozedurebene_names)
    index_tmp = strcmp(prozedurebene_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        prozedurebene_index = index_tmp;
        break
    end
end

prozedur_names = ["Prozedur"];
prozedur_index = [];
for i = 1:length(prozedur_names)
    index_tmp = strcmp(prozedur_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        prozedur_index = index_tmp;
        break
    end
end


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


voltage_names = ["Voltage" , "Spannung" , "Span" ];
voltage_index = [];
for i = 1:length(voltage_names)
    index_tmp = strcmp(voltage_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        voltage_index = index_tmp;
        break
    end
end


current_names = ["Strom","Current"];
current_index = [];
for i = 1:length(current_names)
    index_tmp = strcmp(current_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        current_index = index_tmp;
        break
    end
end

temperature_index = [];


pat = "Temp" + digitsPattern(1)+"5";
k = strfind(spalten_namen,pat);
k(cellfun(@isempty, k)) = {nan};
k = cell2mat(k);
k(isnan(k))=0;
index_tmp = find(k,1,'last');
if ~isempty(index_tmp)
    temperature_index = index_tmp;
end

if isempty(temperature_index)
    pat = "Temp" + digitsPattern(2);
    k = strfind(spalten_namen,pat);
    k(cellfun(@isempty, k)) = {nan};
    k = cell2mat(k);
    k(isnan(k))=0;
    index_tmp = find(k,1,'last');
    if ~isempty(index_tmp)
        temperature_index = index_tmp;
    end
end

if isempty(temperature_index)
    temperature_names = ["Temperatur_","Temperature","Temperatur","Temperatur1"];
    for i = 1:length(temperature_names)
        index_tmp = strcmp(temperature_names(i),spalten_namen);
        index_tmp = find(index_tmp,1);
        if ~isempty(index_tmp)
            temperature_index = index_tmp;
            break
        end
    end
end

if isempty(temperature_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),"Tcan")
            temperature_index = find(contains(spalten_namen,"Tcan"),1);
            break
        end
    end
end

if isempty(temperature_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),"T_Bat")
            temperature_index = find(contains(spalten_namen,"T_Bat"),1);
            break
        end
    end
end

if isempty(temperature_index)
    pat = "Channel" + digitsPattern(3);
    k = strfind(spalten_namen,pat);
    k(cellfun(@isempty, k)) = {nan};
    k = cell2mat(k);
    k(isnan(k))=0;
    index_tmp = find(k,1,'last');
    if ~isempty(index_tmp)
        temperature_index = index_tmp;
    end
end

if isempty(temperature_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),"Temp") && not(contains(spalten_namen(i),"SetTemp")) && not(contains(spalten_namen(i),"ASetTemp"))
            temperature_index = find(contains(spalten_namen,"Temp"),1);
            break
        end
    end
end


if contains(filename, 'J3637_GradiBatt')
    try
        temperatures = mean([diga.daten.TEMP_0_AVG;diga.daten.TEMP_1_AVG],1).';
    catch
        if not(isempty(temperature_index))
            temperatures = diga.daten.(spalten_namen{temperature_index}).';
        else
            temperatures  =[];
        end
    end
else
    if not(isempty(temperature_index))
        temperatures = diga.daten.(spalten_namen{temperature_index}).';
    else
        temperatures  =[];
    end
end


AhAkku_names = ["AhAkku" , "AhStep" ];
AhAkku_index = [];
for i = 1:length(AhAkku_names)
    index_tmp = strcmp(AhAkku_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        AhAkku_index = index_tmp;
        break
    end
end

WhAkku_names = ["WhAkku","WhStep"];
WhAkku_index = [];
for i = 1:length(WhAkku_names)
    index_tmp = strcmp(WhAkku_names(i),spalten_namen);
    index_tmp = find(index_tmp,1);
    if ~isempty(index_tmp)
        WhAkku_index = index_tmp;
        break
    end
end


if isempty(voltage_index)
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
elseif not(isempty(WhAkku_index)) && not(isempty(AhAkku_index)) && not(isempty(temperature_index))
    try
        TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).', ...
            'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS' ...
            ,'TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{voltage_index}).', ...
            diga.daten.(spalten_namen{current_index}).',temperatures, ...
            diga.daten.(spalten_namen{AhAkku_index}).',diga.daten.(spalten_namen{WhAkku_index}).',double(diga.daten.(spalten_namen{schritt_index}).'),categorical(string(diga.daten.(spalten_namen{prozedur_index})).'), ...
            categorical(string(diga.daten.(spalten_namen{zustand_index})).'), double(diga.daten.(spalten_namen{schrittdauer_index}).'), ...
            double(diga.daten.(spalten_namen{zyklus_index}).'), double(diga.daten.(spalten_namen{zyklusebene_index}).'), double(diga.daten.(spalten_namen{prozedurebene_index}).'), ...
            'VariableNames',{'Voltage','Current','Temperature','Ah_Counter','Wh_Counter','Schritt','Prozedur','Zustand','Schrittdauer','Zyklus','Zyklusebene','Prozedurebene'});
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
elseif not(isempty(AhAkku_index)) && not(isempty(temperature_index))
    try
        TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).', ...
            'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS' ...
            ,'TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{voltage_index}).', ...
            diga.daten.(spalten_namen{current_index}).',temperatures, ...
            diga.daten.(spalten_namen{AhAkku_index}).',double(diga.daten.(spalten_namen{schritt_index}).'),categorical(string(diga.daten.(spalten_namen{prozedur_index})).'), ...
            categorical(string(diga.daten.(spalten_namen{zustand_index})).'), double(diga.daten.(spalten_namen{schrittdauer_index}).'), ...
            double(diga.daten.(spalten_namen{zyklus_index}).'), double(diga.daten.(spalten_namen{zyklusebene_index}).'), double(diga.daten.(spalten_namen{prozedurebene_index}).'), ...
            'VariableNames',{'Voltage','Current','Temperature','Ah_Counter','Schritt','Prozedur','Zustand','Schrittdauer','Zyklus','Zyklusebene','Prozedurebene'});
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
elseif not(isempty(temperature_index))
    try
        TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).', ...
            'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS' ...
            ,'TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{voltage_index}).', ...
            diga.daten.(spalten_namen{current_index}).',temperatures,double(diga.daten.(spalten_namen{schritt_index}).'),categorical(string(diga.daten.(spalten_namen{prozedur_index})).'), ...
            categorical(string(diga.daten.(spalten_namen{zustand_index})).'), double(diga.daten.(spalten_namen{schrittdauer_index}).'), ...
            double(diga.daten.(spalten_namen{zyklus_index}).'), double(diga.daten.(spalten_namen{zyklusebene_index}).'), double(diga.daten.(spalten_namen{prozedurebene_index}).'), ...
            'VariableNames',{'Voltage','Current','Temperature','Schritt','Prozedur','Zustand','Schrittdauer','Zyklus','Zyklusebene','Prozedurebene'});
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
else
    try
        TT = timetable(datetime(diga.daten.(spalten_namen{time_index}).', ...
            'ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS' ...
            ,'TimeZone','Europe/Berlin'),diga.daten.(spalten_namen{voltage_index}).', ...
            diga.daten.(spalten_namen{current_index}).',double(diga.daten.(spalten_namen{schritt_index}).'),categorical(string(diga.daten.(spalten_namen{prozedur_index})).'), ...
            categorical(string(diga.daten.(spalten_namen{zustand_index})).'), double(diga.daten.(spalten_namen{schrittdauer_index}).'), ...
            double(diga.daten.(spalten_namen{zyklus_index}).'), double(diga.daten.(spalten_namen{zyklusebene_index}).'), double(diga.daten.(spalten_namen{prozedurebene_index}).'), ...
            'VariableNames',{'Voltage','Current'},'Schritt','Prozedur','Zustand','Schrittdauer','Zyklus','Zyklusebene','Prozedurebene');
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
end


end

