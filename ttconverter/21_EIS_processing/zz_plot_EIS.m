
function zz_plot_EIS( path_to_data, cell_names)
    arguments
        path_to_data string =  "./../../data/"; % configure the path of the data
        cell_names cell =  {[
            "CATL LEP71H3L7_01_004-PROJECT-DISS_abl",...
        ]}; % seperated with "," are plotted together; seperated with ";" seperated from each other

    end

close all;
hide_values = 0;

figure_width_cm = 16;
figure_height_cm = figure_width_cm/3*2;
FontSizePlots = 10;
FontSizeColorbar = 10;
FontSizeTitle = 8;
LineWidthPlots = 1.0;
FontNamePlots = "arial";
title_name = "";

set(0,'defaulttextinterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

save_plots = 0;



% % seperated with "," are plotted together; seperated with ";" seperated from each other
% cell_names = {["LiFun_575166-01_002-PROJECT-J3647_BMBF_OSLiB"]}; 


Current_limits = [-1000, 1000];
Temperature_limits = [-100,100]; %[-100, -20; -20, 20; 20, 100]
Voltage_limits = [0.0, 5.0]; %[3.6, 3.8; 3.8, 5.0]
Time_limits = datetime(["1900-11-19 09:00:00.000000", "2090-01-01 00:00:00.000000"],'TimeZone','Europe/Berlin');
Ah_throughput_limits = [-100, 10000000];
Wh_throughput_limits = [-100, 10000000];
Capacity_limits = [-1000, 1000];
Capacity_current_limits = [-1000, 1000];
SOH_limits = [-1000, 1000];
SOC_limits = [-1000, 1000];
Frequency_limits = [-10, 100000];

%'Current'	'Temperature'	'Voltage'	'Duration'	'Ah_throughput'
%'Wh_throughput'	'Capacity'	'Capacity_current'	'SOH'	'SOC'
variable_for_color = 'Voltage'; 

global_limits = 0;



[cell_name_size_y,cell_name_size_x] = cellfun(@size,cell_names);

% if max(cell_name_size_x)>1
%     global_limits = 1;
% end


variable_name = {'Time','Current', 'Temperature', 'Voltage', 'Duration', 'Ah_throughput', 'Wh_throughput', 'Capacity', 'Capacity_current', 'SOH', 'SOC', 'SOC_py'};
variable_unit = ["days", "A"  "^\circ C"  "V"  "days"  "Ah"  "Wh"  "Ah"  "A"  "\%"  "\%" "\%"];
variable_for_color_label = containers.Map(variable_name,variable_unit);

color_steps_quantisation = 10000;

if strcmp(variable_for_color,'SOH') | strcmp(variable_for_color,'SOC') | strcmp(variable_for_color,'Voltage')  | strcmp(variable_for_color,'Capacity')
    mycolor = flipud(turbo(color_steps_quantisation));
else
    mycolor = turbo(color_steps_quantisation);
end

% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"export/"));
mkdir(strcat(path_to_data,"figures/"));
mkdir(strcat(path_to_data,"eis_data/"));



% global plot limits
% calculate all Z
if global_limits == 1
    TT_eis_all = timetable();
    TT_eis_all.Time.TimeZone = 'Europe/Berlin';
    TT_eis_all.Time.Format = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    for cell_index_global = 1:size(cell_names,1)
        for cell_index = 1:size(cell_names{cell_index_global},2)
            % load the EIS timeseries 
            try
                load(strcat(path_to_data,'eis_data\',cell_names{cell_index_global}{cell_index},'_eis.mat'),'TT_eis');
                TT_eis.EIS_Frequency(1); % test if data is accessible
                TT_eis_all = [TT_eis_all;TT_eis];
            catch
                error(strcat("EIS timeseries file could not be load: ", strcat(path_to_data,'eis_data\',cell_names{cell_index_global}{cell_index},'_eis.mat') ));
                return;
            end
            
        end
    end

    TT_eis_all.SOH = TT_eis_all.SOH*100;
    TT_eis_all.SOC = TT_eis_all.SOC*100;

    Z_all = TT_eis_all.EIS_Z_abs(:) .* exp(1i* TT_eis_all.EIS_Z_phase(:));

    nyquist_x_lim = [min(real(Z_all)) , max(real(Z_all))].*1000;
    nyquist_y_lim = [min(imag(Z_all)) , max(imag(Z_all))].*1000;
    bode_amp_y_lim = [min(abs(Z_all)) , max(abs(Z_all))].*1000;
    bode_phase_y_lim = [min(angle(Z_all)/pi*180) , max(angle(Z_all)/pi*180)];
    bode_freq_x_lim = [min(TT_eis_all.EIS_Frequency) , max(TT_eis_all.EIS_Frequency)];

    if strcmp(variable_for_color, 'Time')
        color_lim = days([min(TT_eis_all.Time), max(TT_eis_all.Time)] - min(TT_eis_all.Time));
    else
        color_lim = [min(TT_eis_all{:,variable_for_color}), max(TT_eis_all{:,variable_for_color})];
    end
    color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);

    clear TT_eis_all
end

fig_counter = 1;

% plott for all combinations of limits
for cell_index_global = 1:size(cell_names,1)
    if max(cell_name_size_x)>1
        fig_global = figure(fig_counter);
        fig_counter = fig_counter + 1;
        if strlength(strjoin(cell_names{cell_index_global})) < 100
            fig_global.Name = strcat("Global: ",strjoin(cell_names{cell_index_global}));
        else
            fig_global.Name = strcat("Global: All Cells");
        end
        set(fig_global,'Units','centimeters')
        set(fig_global,'Position',[1,1,figure_width_cm,figure_height_cm]);
        tLayout_global = tiledlayout(2,3);
    end
    for cell_index = 1:size(cell_names{cell_index_global},2)
        for voltage_index = 1:size(Voltage_limits,1)
            for temperature_index = 1:size(Temperature_limits,1)
                for current_index = 1:size(Current_limits,1)
                    for ah_lindex = 1:size(Ah_throughput_limits,1)
                        for wh_lindex = 1:size(Wh_throughput_limits,1)
                            for time_index = 1:size(Time_limits,1)
                                for capacity_index = 1:size(Capacity_limits,1)
                                    for capacity_cur_index = 1:size(Capacity_current_limits,1)
                                        for soh_index = 1:size(SOH_limits,1)
                                            for soc_index = 1:size(SOC_limits,1)
                                                for frequency_index = 1:size(Frequency_limits,1)
    
                                                    % load the EIS timeseries 
                                                    try
                                                        load(strcat(path_to_data,'eis_data\',cell_names{cell_index_global}{cell_index},'_eis.mat'),'TT_eis');
                                                        TT_eis.EIS_Frequency(1); % test if data is accessible

                                                        TT_eis.SOH(isnan(TT_eis.SOH)) = -42;
                                                        TT_eis.SOC(isnan(TT_eis.SOC)) = -42;
                                                        TT_eis.Capacity(isnan(TT_eis.Capacity)) = -42;
                                                        TT_eis.Capacity_current(isnan(TT_eis.Capacity_current)) = -42;
                                                        try
                                                            TT_eis = removevars(TT_eis,'Prozedur');
                                                        end
                                                        try
                                                            TT_eis = removevars(TT_eis,'Zustand');
                                                        end
                                                        try
                                                            TT_eis = removevars(TT_eis,'Pulse_Name');
                                                        end
                                                        try
                                                            TT_eis = removevars(TT_eis,'Name_Ahjo');
                                                        end
                                                        try
                                                            TT_eis = removevars(TT_eis,'Capacity_Name');
                                                        end

                                                        % TT_eis = fillmissing(TT_eis,'constant',-1);
                                                    catch
                                                        error(strcat("EIS timeseries file could not be load: ", strcat(path_to_data,'eis_data\',cell_names{cell_index_global}{cell_index},'_eis.mat') ));
                                                        return;
                                                    end
    
                                                    TT_eis_sub = TT_eis(    ...
                                                        TT_eis.Wh_throughput    >       Wh_throughput_limits(wh_lindex,1)               &           TT_eis.Wh_throughput    <           Wh_throughput_limits(wh_lindex,2)               &...
                                                        TT_eis.Capacity         >       Capacity_limits(capacity_index,1)               &           TT_eis.Capacity         <           Capacity_limits(capacity_index,2)               &...
                                                        TT_eis.Capacity_current >       Capacity_current_limits(capacity_cur_index,1)   &           TT_eis.Capacity_current <           Capacity_current_limits(capacity_cur_index,2)   &...
                                                        TT_eis.SOH              >       SOH_limits(soh_index,1)                         &           TT_eis.SOH              <           SOH_limits(soh_index,2)                         &...
                                                        TT_eis.SOC              >       SOC_limits(soc_index,1)                         &           TT_eis.SOC              <           SOC_limits(soc_index,2)                         &...
                                                        TT_eis.Voltage          >       Voltage_limits(voltage_index,1)                 &           TT_eis.Voltage          <           Voltage_limits(voltage_index,2)                 &...
                                                        TT_eis.Temperature      >       Temperature_limits(temperature_index,1)         &           TT_eis.Temperature      <           Temperature_limits(temperature_index,2)         &...
                                                        TT_eis.Current          >       Current_limits(current_index,1)                 &           TT_eis.Current          <           Current_limits(current_index,2)                 &...
                                                        TT_eis.Ah_throughput    >       Ah_throughput_limits(ah_lindex,1)               &           TT_eis.Ah_throughput    <           Ah_throughput_limits(ah_lindex,2)               &...
                                                        TT_eis.Time             >       Time_limits(time_index,1)                       &           TT_eis.Time             <           Time_limits(time_index,2)                       &...
                                                        TT_eis.EIS_Frequency    >       Frequency_limits(frequency_index,1)             &           TT_eis.EIS_Frequency    <           Frequency_limits(frequency_index,2)             ...
                                                    ,:);
    %                                                 check if subset is empty
                                                    if isempty(TT_eis_sub)
                                                        continue;
                                                    end
    
                                                    TT_eis_sub.SOH = TT_eis_sub.SOH*100;
                                                    TT_eis_sub.SOC = TT_eis_sub.SOC*100;
    
    %                                                 if global limits are not set, calculate local ones
                                                    if not(global_limits == 1)
                                                        Z_all = TT_eis_sub.EIS_Z_abs(:) .* exp(1i* TT_eis_sub.EIS_Z_phase(:));
                            
    %                                                     change to milliohm
                                                        Z_all = Z_all.*1000;
                            
                                                        nyquist_x_lim = [min(real(Z_all)) , max(real(Z_all))];
                                                        nyquist_y_lim = [min(imag(Z_all)) , max(imag(Z_all))];
                                                        bode_amp_y_lim = [min(abs(Z_all)) , max(abs(Z_all))];
                                                        bode_phase_y_lim = [min(angle(Z_all)/pi*180) , max(angle(Z_all)/pi*180)];
                                                        bode_freq_x_lim = [min(TT_eis_sub.EIS_Frequency) , max(TT_eis_sub.EIS_Frequency)];
                            
                                                        if strcmp(variable_for_color, 'Time')
                                                            color_lim = days([min(TT_eis_sub.Time), max(TT_eis_sub.Time)] - min(TT_eis_sub.Time));
                                                        else
                                                            color_lim = [min(TT_eis_sub{:,variable_for_color}), max(TT_eis_sub{:,variable_for_color})];
                                                        end
                                                        color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);
                                                    end
                            
                                                    measurements = unique(TT_eis_sub.EIS_measurement_id);
                                                    
                                                    fig = figure(fig_counter);
                                                    fig_counter = fig_counter + 1;
                                                    fig.Name = strcat(cell_names{cell_index_global}{cell_index});
                                                    set(fig,'Units','centimeters')
                                                    set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
                                                    hold on;


                                                    sgtitle(title_name,'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    tLayout = tiledlayout(2,3);
                            
                                                    for measurement_index = 1:length(measurements)
                                                        TT_eis_sub_sub = TT_eis_sub(TT_eis_sub.EIS_measurement_id == measurements(measurement_index) ,:);
                            
    %                                                     voltage_plot = mean(TT_eis_sub_sub.Voltage);
    %                                                     temperature_plot = mean(TT_eis_sub_sub.Temperature);
    %                                                     current_plot = mean(TT_eis_sub_sub.Current);
    %                                                     ah_plot = mean(TT_eis_sub_sub.Ah_throughput);
    %                                                     time_plot = mean(TT_eis_sub_sub.Time);
    
                                                        frequency_plot = TT_eis_sub_sub.EIS_Frequency;
                                                        z_plot = TT_eis_sub_sub.EIS_Z_abs(:) .* exp(1i* TT_eis_sub_sub.EIS_Z_phase(:));
                            
                            
                            
                                                        if variable_for_color == "Time"
                                                            color_plot = days(mean(TT_eis_sub_sub.Time) - min(TT_eis.Time));
                                                        else
                                                            color_plot = mean(TT_eis_sub_sub{:,variable_for_color});
                                                        end
                            
                                                        [minValue, colorIndex] = min(abs(color_steps-color_plot));
                                                        
                            
                                                        %change to milli Ohm
                                                        z_plot = z_plot.*1000;


                                                        set(0, 'CurrentFigure', fig);
                                                        % Nyquist
                                                        nexttile(1, [2 1]);
                                                        hold on
                                                        plot(real(z_plot), imag(z_plot), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
                            
                                                        % Bode Absolute
                                                        nexttile(2, [1 2]);
                                                        hold on
                                                        plot(frequency_plot,abs(z_plot), 'color', mycolor(colorIndex, :));
                            
                                                        % Bode Phase
                                                        nexttile(5, [1 2]);
                                                        hold on
                                                        plot(frequency_plot,angle(z_plot)/pi*180, 'color', mycolor(colorIndex, :));
                                                        


                                                        % Global Plot now
                                                        if max(cell_name_size_x)>1
                                                            set(0, 'CurrentFigure', fig_global);
                                                            % Nyquist
                                                            nexttile(1, [2 1]);
                                                            hold on
                                                            plot(real(z_plot), imag(z_plot), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
                                
                                                            % Bode Absolute
                                                            nexttile(2, [1 2]);
                                                            hold on
                                                            plot(frequency_plot,abs(z_plot), 'color', mycolor(colorIndex, :));
                                
                                                            % Bode Phase
                                                            nexttile(5, [1 2]);
                                                            hold on
                                                            plot(frequency_plot,angle(z_plot)/pi*180, 'color', mycolor(colorIndex, :));
                                                        end
                                                    end
                            
                                                    set(0, 'CurrentFigure', fig);
                                                    % Nyquist
                                                    tile_nyquist = nexttile(1,[2 1]);
                                                    hold on
                                                    set(gca,'FontSize',FontSizePlots)
                                                    set(gca,'FontName',FontNamePlots)
                                                    axis(gca,'equal')
                                                    set(gca, 'YDir','reverse');
                                                    xlabel(tile_nyquist,'$\Re(\underline{Z})$ in m$\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    ylabel(tile_nyquist,'$\Im(\underline{Z})$ in m$\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    if diff(color_lim) > 0
                                                        colorbar_plot = colorbar;
                                                        if strcmp(variable_for_color,'SOH') | strcmp(variable_for_color,'SOC') | strcmp(variable_for_color,'Voltage')  | strcmp(variable_for_color,'Capacity')
                                                            %'Current'	'Temperature'	'Voltage'	'Duration'	'Ah_throughput'
                                                            %'Wh_throughput'	'Capacity'	'Capacity_current'	'SOH'	'SOC'
                                                            colorbar_plot.Direction = 'reverse';
                                                        end
                                                        colorbar_plot.Label.Interpreter = 'latex';
                                                        colorbar_plot.Label.String = strcat("$\mathrm{",replace(variable_for_color,"_"," "), "\ in\ ", variable_for_color_label(variable_for_color),"}$");
                                                        colorbar_plot.FontSize = FontSizeTitle;
                                                        colorbar_plot.Label.FontSize = FontSizeColorbar;
                                                        colorbar_plot.Label.FontName = FontNamePlots;
                                    %                     colorbar_plot.Location = 'northoutside';
                                                        colorbar_plot.Layout.TileSpan = [1 1];
                                                        colorbar_plot.Layout.Tile = 'north';
                                                    end
                                                    grid on
                                                    colormap(mycolor)
                                                    if diff(color_lim) > 0
                                                        clim(color_lim)
                                                    end
                                                    xlim(nyquist_x_lim);
                                                    ylim(nyquist_y_lim);
                            
                            
                                                    % Bode Absolute
                                                    tile_bode_abs = nexttile(2, [1 2]);
                                                    hold on
                                                    set(gca,'FontSize',FontSizePlots)
                                                    set(gca,'FontName',FontNamePlots)
                                                    set(gca,'Xscale','log');
    
                                                    ylabel(tile_bode_abs,'$|\underline{Z}|$ in m$\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    grid on
                                                    colormap(mycolor)
                                                    if diff(color_lim) > 0
                                                        clim(color_lim)
                                                    end
                                                    xlim(bode_freq_x_lim);
                                                    ylim(bode_amp_y_lim);
                            
                                                    % Bode Phase
                                                    tile_bode_ang = nexttile(5, [1 2]);
    
                                                    set(gca,'FontSize',FontSizePlots)
                                                    set(gca,'FontName',FontNamePlots)
                                                    set(gca,'Xscale','log');
    
                                                    xlabel(tile_bode_ang,'Frequency in Hz','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    ylabel(tile_bode_ang,'\angle $\underline{Z}$ in $^\circ$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                                                    grid on
                                                    colormap(mycolor)
                                                    if diff(color_lim) > 0
                                                        clim(color_lim)
                                                    end
                                                    xlim(bode_freq_x_lim);
                                                    ylim(bode_phase_y_lim);
                            
                                                    
    %                                                 generic settings
                                                    tLayout.TileSpacing = 'compact';
                                                    tLayout.Padding = 'compact';
                            
                                                    file_name = strcat(replace(cell_names{cell_index_global}{cell_index}," ","_"));
                                                    file_name = strcat(file_name,'_',variable_for_color);
                                                    save_path = strcat(path_to_data,"figures\",file_name);
                            
    
                                                    if hide_values
                                                        nexttile(1,[2 1]);
    %                                                     set(gca,'xtick',[])
                                                        set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
                                                        set(gca,'yticklabel',[])
                                                        nexttile(2, [1 2]);
    %                                                     set(gca,'xtick',[])
    %                                                     set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
                                                        set(gca,'yticklabel',[])
                                                        nexttile(5, [1 2]);
    %                                                     set(gca,'xtick',[])
    %                                                     set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
                                                        set(gca,'yticklabel',[])
                                                    end
                            
                            
                            
                                                    if save_plots
                                                        drawnow;
                            
                                                        if exist('exportgraphics')
                                                            set(gcf, 'color', 'none'); 
                                                            nexttile(1,[2 1]);
                                                            set(gca, 'color', 'none');
                                                            nexttile(2, [1 2]);
                                                            set(gca, 'color', 'none');
                                                            nexttile(5, [1 2]);
                                                            set(gca, 'color', 'none');
    %                                                         savefig(strcat(save_path,'.fig'));
    %                                                         exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %                                                         exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %                                                         exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                                                            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
                                                        else
    %                                                         savefig(strcat(save_path,'.fig'));
    %                                                         saveas(gcf,save_path,'svg');
    %                                                         saveas(gcf,save_path,'pdf');
    %                                                         saveas(gcf,save_path,'emf');
                                                            saveas(gcf,save_path,'png');
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    if max(cell_name_size_x)>1
        if strlength(replace(strjoin(cell_names{cell_index_global})," ","_")) < 100
            global_file_name = replace(strjoin(cell_names{cell_index_global})," ","_");
        else
            global_file_name = "all_cells";
        end
        global_file_name = strcat(global_file_name,'_',variable_for_color);
        set(0, 'CurrentFigure', fig_global);
        % Nyquist
        tile_nyquist = nexttile(1,[2 1]);
        hold on
        set(gca,'FontSize',FontSizePlots)
        set(gca,'FontName',FontNamePlots)
        axis(gca,'equal')
        set(gca, 'YDir','reverse');
        xlabel(tile_nyquist,'$\Re(\underline{Z})$ in m$\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        ylabel(tile_nyquist,'$\Im(\underline{Z})$ in m$\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        if diff(color_lim) > 0
            colorbar_plot = colorbar;
            if strcmp(variable_for_color,'SOH') | strcmp(variable_for_color,'SOC') | strcmp(variable_for_color,'Voltage')  | strcmp(variable_for_color,'Capacity')
                %'Current'	'Temperature'	'Voltage'	'Duration'	'Ah_throughput'
                %'Wh_throughput'	'Capacity'	'Capacity_current'	'SOH'	'SOC'
                colorbar_plot.Direction = 'reverse';
            end
            colorbar_plot.Label.Interpreter = 'latex';
            colorbar_plot.Label.String = strcat("$\mathrm{",replace(variable_for_color,"_"," "), "\ in\ ", variable_for_color_label(variable_for_color),"}$");
            colorbar_plot.FontSize = FontSizeTitle;
            colorbar_plot.Label.FontSize = FontSizeColorbar;
            colorbar_plot.Label.FontName = FontNamePlots;
    %                     colorbar_plot.Location = 'northoutside';
            colorbar_plot.Layout.TileSpan = [1 1];
            colorbar_plot.Layout.Tile = 'north';
        end
        grid on
        colormap(mycolor)
        if diff(color_lim) > 0
            caxis(color_lim)
        end
        if global_limits == 1
            xlim(nyquist_x_lim);
            ylim(nyquist_y_lim);
        end
    
        % Bode Absolute
        tile_bode_abs = nexttile(2, [1 2]);
        hold on
        set(gca,'FontSize',FontSizePlots)
        set(gca,'FontName',FontNamePlots)
        set(gca,'Xscale','log');
    
        ylabel(tile_bode_abs,'$|\underline{Z}|$ in $\Omega$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        grid on
        colormap(mycolor)
        if diff(color_lim) > 0
            caxis(color_lim)
        end
        if global_limits == 1
            xlim(bode_freq_x_lim);
            ylim(bode_amp_y_lim);
        end
    
        % Bode Phase
        tile_bode_ang = nexttile(5, [1 2]);
    
        set(gca,'FontSize',FontSizePlots)
        set(gca,'FontName',FontNamePlots)
        set(gca,'Xscale','log');
    
        xlabel(tile_bode_ang,'Frequency in Hz','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        ylabel(tile_bode_ang,'\angle $\underline{Z}$ in $^\circ$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        grid on
        colormap(mycolor)
        if diff(color_lim) > 0
            caxis(color_lim)
        end
        if global_limits == 1
            xlim(bode_freq_x_lim);
            ylim(bode_phase_y_lim);
        end
    
        
    %                                                 generic settings
        tLayout.TileSpacing = 'compact';
        tLayout.Padding = 'compact';
    
        save_path = strcat(path_to_data,"figures\",global_file_name);
    
    
        if hide_values
            nexttile(1,[2 1]);
    %                                                     set(gca,'xtick',[])
            set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            nexttile(2, [1 2]);
    %                                                     set(gca,'xtick',[])
    %                                                     set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
            set(gca,'yticklabel',[])
            nexttile(5, [1 2]);
    %                                                     set(gca,'xtick',[])
    %                                                     set(gca,'xticklabel',[])
    %                                                     set(gca,'ytick',[])
            set(gca,'yticklabel',[])
        end
    
    
    
        if save_plots
            drawnow;
    
            if exist('exportgraphics')
                set(gcf, 'color', 'none'); 
                nexttile(1,[2 1]);
                set(gca, 'color', 'none');
                nexttile(2, [1 2]);
                set(gca, 'color', 'none');
                nexttile(5, [1 2]);
                set(gca, 'color', 'none');
%                 savefig(strcat(save_path,'.fig'));
%                 exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
%                 exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
%                 exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
            else
%                 savefig(strcat(save_path,'.fig'));
%                 saveas(gcf,save_path,'svg');
%                 saveas(gcf,save_path,'pdf');
%                 saveas(gcf,save_path,'emf');
                saveas(gcf,save_path,'png');
            end
        end
    end
end

end
