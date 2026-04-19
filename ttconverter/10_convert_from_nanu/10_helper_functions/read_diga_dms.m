function TT = read_diga_dms(filename)
%READ_DIGA Summary of this function goes here
%   Detailed explanation goes here

if contains(filename,"invalid",'IgnoreCase',true)
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


dms_names = "DMS";
dms_index = [];
if isempty(dms_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),dms_names)
            dms_index = find(contains(spalten_namen,dms_names),1);
            break
        end
    end
end

kraft_names = "Kraft";
kraft_index = [];
if isempty(kraft_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),kraft_names)
            kraft_index = find(contains(spalten_namen,kraft_names),1);
            break
        end
    end
end

messuhr_names = "Messuhr";
messuhr_index = [];
if isempty(messuhr_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),messuhr_names)
            messuhr_index = find(contains(spalten_namen,messuhr_names),1);
            break
        end
    end
end

feinzeiger_names = "Feinzeiger";
feinzeiger_index = [];
if isempty(feinzeiger_index)
    for i = 1:length(spalten_namen)
        if contains(spalten_namen(i),feinzeiger_names)
            feinzeiger_index = find(contains(spalten_namen,feinzeiger_names),1);
            break
        end
    end
end


if isempty(dms_index) && isempty(kraft_index) && isempty(messuhr_index) && isempty(feinzeiger_index)
    TT = timetable();
    TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    TT.Time.TimeZone = 'Europe/Berlin';
else
    try
        time = datetime(diga.daten.(spalten_namen{time_index}).','ConvertFrom','posixtime','Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin');
        nan_columns = nan(length(time),1);
        if not(isempty(dms_index))
            dms_values = diga.daten.(spalten_namen{dms_index}).';
            tt_dms = timetable(time,dms_values,'VariableNames',{'DMS'});
            tt_dms = convertvars(tt_dms,{'DMS'},'double');
        else
            tt_dms = timetable(time,nan_columns,'VariableNames',{'DMS'});
            tt_dms = convertvars(tt_dms,{'DMS'},'double');
        end

        if not(isempty(kraft_index))
            kraft_values = diga.daten.(spalten_namen{kraft_index}).';
            tt_kraft = timetable(time,kraft_values,'VariableNames',{'Kraft'});
            tt_kraft = convertvars(tt_kraft,{'Kraft'},'double');
        else
            tt_kraft = timetable(time,nan_columns,'VariableNames',{'Kraft'});
            tt_kraft = convertvars(tt_kraft,{'Kraft'},'double');
        end

        if not(isempty(messuhr_index))
            messuhr_values = diga.daten.(spalten_namen{messuhr_index}).';
            tt_messuhr = timetable(time,messuhr_values,'VariableNames',{'Messuhr'});
            tt_messuhr = convertvars(tt_messuhr,{'Messuhr'},'double');
        else
            tt_messuhr = timetable(time,nan_columns,'VariableNames',{'Messuhr'});
            tt_messuhr = convertvars(tt_messuhr,{'Messuhr'},'double');
        end

        if not(isempty(feinzeiger_index))
            feinzeiger_values = diga.daten.(spalten_namen{feinzeiger_index}).';
            tt_feinzeiger = timetable(time,feinzeiger_values,'VariableNames',{'Feinzeiger'});
            tt_feinzeiger = convertvars(tt_feinzeiger,{'Feinzeiger'},'double');
        else
            tt_feinzeiger = timetable(time,nan_columns,'VariableNames',{'Feinzeiger'});
            tt_feinzeiger = convertvars(tt_feinzeiger,{'Feinzeiger'},'double');
        end
        
        TT = [tt_dms,tt_kraft,tt_messuhr,tt_feinzeiger];
    catch
        TT = timetable();
        TT.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
        TT.Time.TimeZone = 'Europe/Berlin';
    end
end

end

