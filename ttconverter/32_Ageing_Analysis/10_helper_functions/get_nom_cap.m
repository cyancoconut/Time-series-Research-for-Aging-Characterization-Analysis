function T_nom_cap = get_nom_cap(Cell_Name,path_to_ahjo_cap_data_table)


    opts = spreadsheetImportOptions("NumVariables", 2);
    
    % Specify sheet and range
    opts.Sheet = "Tabelle1";
    opts.DataRange = "A2:B1048576";
    
    % Specify column names and types
    opts.VariableNames = ["name", "Capacity"];
    opts.VariableTypes = ["string", "string"];
    
    % Specify variable properties
    opts = setvaropts(opts, ["name", "Capacity"], "WhitespaceRule", "preserve");
    opts = setvaropts(opts, ["name", "Capacity"], "EmptyFieldRule", "auto");
    
    % Import the data
    Ahjospecimencap = readtable(path_to_ahjo_cap_data_table, opts, "UseExcel", false);
    
    T_nom_cap = Ahjospecimencap(ismember(Ahjospecimencap.name,Cell_Name.("Cell Name")),:);
    T_nom_cap.Properties.VariableNames={'Cell Name','Nominal Capacity in Ah'};
    
    T_nom_cap = outerjoin(Cell_Name,T_nom_cap,'MergeKeys',true);
    T_nom_cap.("Nominal Capacity in Ah") = str2double(replace(T_nom_cap.("Nominal Capacity in Ah"),',','.'));
end