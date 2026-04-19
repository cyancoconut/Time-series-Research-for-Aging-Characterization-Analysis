function x_plot_ocv(varargin)
    tic
    close all;
    Cell_limits = 1;
    Cell_index = 1;
    hide_values = 0;
    only_most_frequent_capacity = 0;
    hide_name =0;
    
    figure_width_cm = 26;
    figure_height_cm = figure_width_cm/2;
    FontSizePlots = 10;
    FontNamePlots = "arial";
    title_name = "";
    
    cell_name = "LiFun_575166-01_002-PROJECT-J3647_BMBF_OSLiB";
    path_to_data = "./../../data/";
    save_plots = 1;
    
    Voltage_limits = [0.0, 5.0]; %[3.6, 3.8; 3.8, 5.0]
    Temperature_limits = [-100,100]; %[-100, -20; -20, 20; 20, 100]
    Current_limits = [-5, 5];
    Ah_throughput_limits = [-100, 10000000];
    Time_limits = datetime(["1990-01-01 00:00:00.000000", "2090-01-01 00:00:00.000000"],'TimeZone','Europe/Berlin');
    Frequency_limits = [-10, 10000];
    
    variable_for_color = 'Current'; % 'Time', 'Current', 'Voltage', 'Temperature', 'EIS_Frequency', 'EIS_Z_abs', 'EIS_Z_phase', 'Ah_throughput', 'EIS_measurement_id'
    color_steps_quantisation = 1000;
    mycolor = jet(color_steps_quantisation); 
    global_limits = false;
    
    % add helper functions to path
    addpath("./10_helper_functions");

    %% config arguments
    for i = 1:2:length(varargin)
        eval(strcat(varargin{i},"=",varargin{i+1},";"))
    end
    
    % create all necessary folders normally they should already be there
    mkdir(strcat(path_to_data,"figures/"));
    
    title_name = replace(cell_name,'_',' ');
    
    
    % load the EIS timeseries 
    try
        load(strcat(path_to_data,'ocv_data\',cell_name,'_ocv.mat'),'TT_ocv');
        TT_ocv.Capacity(1); % test if data is accessible
    catch
        error(strcat("OCV timeseries file could not be load: ", strcat(path_to_data,'ocv_data\',cell_name,'_ocv.mat') ));
        return;
    end
    
    TT_ocv.Temperature( TT_ocv.Temperature < -50) = NaN;
    
    
    % search for most used current at capacity tests and only use those
    % "automatic" current limiting
    if only_most_frequent_capacity
        [OCV_m_id,OCV_m_id_index] = unique(TT_ocv.OCV_measurement_id);
        currents = TT_ocv(OCV_m_id_index,"Capacity_current");
        currents = currents.Capacity_current;
        currents(currents>0) = nan;
        % peak find
        try
            [currents_N,currents_edges] = histcounts(currents);
            [~,index_edges] = max(currents_N);
            currents_min = currents_edges(index_edges);
            currents_max = currents_edges(index_edges+1);
        end
    
        OCV_m_id_index_table = TT_ocv(OCV_m_id_index,:);
        OCV_m_id_index_table = OCV_m_id_index_table(OCV_m_id_index_table.Capacity_current > currents_min &  OCV_m_id_index_table.Capacity_current < currents_max,:);

        OCV_m_id_valid_pos = unique(OCV_m_id_index_table.OCV_measurement_id);


        [OCV_m_id,OCV_m_id_index] = unique(TT_ocv.OCV_measurement_id);
        currents = TT_ocv(OCV_m_id_index,"Capacity_current");
        currents = currents.Capacity_current;
        currents(currents<0) = nan;
        % peak find
        try
            [currents_N,currents_edges] = histcounts(currents);
            [~,index_edges] = max(currents_N);
            currents_min = currents_edges(index_edges);
            currents_max = currents_edges(index_edges+1);
        end
    
        OCV_m_id_index_table = TT_ocv(OCV_m_id_index,:);
        OCV_m_id_index_table = OCV_m_id_index_table(OCV_m_id_index_table.Capacity_current > currents_min &  OCV_m_id_index_table.Capacity_current < currents_max,:);

        OCV_m_id_valid_neg = unique(OCV_m_id_index_table.OCV_measurement_id);

        TT_ocv = TT_ocv(ismember(TT_ocv.OCV_measurement_id,[OCV_m_id_valid_neg;OCV_m_id_valid_pos]) ,:);
    end
    
    if strcmp(variable_for_color, 'Time')
        color_lim = days([min(TT_ocv.Time), max(TT_ocv.Time)] - min(TT_ocv.Time));
    else
        color_lim = [min(TT_ocv{:,variable_for_color}), max(TT_ocv{:,variable_for_color})];
    end
    
    % plott for all combinations of limits
    for voltage_index = 1:size(Voltage_limits,1)
        for temperature_index = 1:size(Temperature_limits,1)
            for current_index = 1:size(Current_limits,1)
                for ah_lindex = 1:size(Ah_throughput_limits,1)
                    for time_index = 1:size(Time_limits,1)
                        TT_ocv_sub = TT_ocv(    TT_ocv.Voltage          >       Voltage_limits(voltage_index,1)             &           TT_ocv.Voltage          <           Voltage_limits(voltage_index,2)             &...
                                                TT_ocv.Temperature      >       Temperature_limits(temperature_index,1)     &           TT_ocv.Temperature      <           Temperature_limits(temperature_index,2)     &...
                                                TT_ocv.Current          >       Current_limits(current_index,1)             &           TT_ocv.Current          <           Current_limits(current_index,2)             &...
                                                TT_ocv.Ah_throughput    >       Ah_throughput_limits(ah_lindex,1)           &           TT_ocv.Ah_throughput    <           Ah_throughput_limits(ah_lindex,2)           &...
                                                TT_ocv.Time             >       Time_limits(time_index,1)                   &           TT_ocv.Time             <           Time_limits(time_index,2)                   ...
                                            ,:);
    
    %                     check if subset is empty
                        if isempty(TT_ocv_sub)
                            continue;
                        end
    
    
                        measurements = unique(TT_ocv_sub.OCV_measurement_id);
    
                        fig = figure(1+(size(Cell_limits,1)-1)*Cell_index + (size(Voltage_limits,1)-1)*voltage_index+(size(Temperature_limits,1)-1)*temperature_index+(size(Current_limits,1)-1)*current_index+(size(Ah_throughput_limits,1)-1)*ah_lindex+(size(Time_limits,1)-1)*time_index);
                        set(fig,'Units','centimeters')
                        set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
                        hold on;
                        if hide_name == 0
                            sgtitle(title_name,'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        end
                        subp = [subplot(2,3,[1 2 3]) subplot(2,3,[4 5 6]) ];
    
                        for measurement_index = 1:length(measurements)
                            TT_ocv_sub_sub = TT_ocv_sub(TT_ocv_sub.OCV_measurement_id == measurements(measurement_index) ,:);
    
                            voltage_plot = mean(TT_ocv_sub_sub.Voltage);
                            temperature_plot = mean(TT_ocv_sub_sub.Temperature);
                            current_plot = mean(TT_ocv_sub_sub.Current);
                            ah_plot = mean(TT_ocv_sub_sub.Ah_throughput);
                            time_plot = mean(TT_ocv_sub_sub.Time);
                            capacity_plot = mean(TT_ocv_sub_sub.Capacity);
                            capacity_current_plot = mean(TT_ocv_sub_sub.Capacity_current);
    
                            capacity = TT_ocv_sub_sub.Ah_Counter;
                            capacity = capacity - min(capacity);
                            capacity = capacity.*sign(TT_ocv_sub_sub.Current);
                            
                            capacity = capacity - min(capacity);
    
                            voltage = TT_ocv_sub_sub.Voltage;
    
                            capacity_charge = capacity(TT_ocv_sub_sub.Current>0);
                            capacity_discharge = capacity(TT_ocv_sub_sub.Current<0);
                            
                            capacity_discharge = capacity_discharge - max(capacity_discharge);
                            capacity_discharge = -capacity_discharge;

    
                            voltage_charge = voltage(TT_ocv_sub_sub.Current>0);
                            voltage_discharge = voltage(TT_ocv_sub_sub.Current<0);
    
                            if variable_for_color == "Time"
                                color_plot = days(mean(TT_ocv_sub_sub.Time) - min(TT_ocv.Time));
                            else
                                color_plot = mean(TT_ocv_sub_sub{:,variable_for_color});
                            end
                            color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);
                            [minValue, colorIndex] = min(abs(color_steps-color_plot));
    
    
                            % discharge: Voltage over capacity
                            set(subp(2),'NextPlot','add');
                            set(fig,'CurrentAxes',subp(2));
                            plot(subp(2),capacity_discharge, voltage_discharge, 'color', mycolor(colorIndex, :));
    
                            % charge: Voltage over capacity
                            set(subp(1),'NextPlot','add');
                            set(fig,'CurrentAxes',subp(1));
                            plot(subp(1),capacity_charge, voltage_charge, 'color', mycolor(colorIndex, :));
                        end
    
    
                        % OCV over Time
                        set(subp(2),'NextPlot','add');
                        set(fig,'CurrentAxes',subp(2));
                        xlabel('Capacity in Ah','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        ylabel('Voltage in V','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        title('Discharge','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        colorbar_plot = colorbar;
                        colorbar_plot.Label.String = replace(variable_for_color,"_"," ");
                        colorbar_plot.Label.Interpreter = 'latex';
                        colorbar_plot.Label.FontSize = FontSizePlots;
                        colorbar_plot.Label.FontName = FontNamePlots;
                        set(gca,'FontSize',FontSizePlots)
                        set(gca,'FontName',FontNamePlots)
                        grid on
                        colormap(mycolor)   
                        caxis(color_lim)
    %                     xlim(ocv_x_lim);
    %                     ylim(ocv_y_lim);
%                         set(gca, 'XDir','reverse');
    
                        set(subp(1),'NextPlot','add');
                        set(fig,'CurrentAxes',subp(1));
                        xlabel('Capacity in Ah','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        ylabel('Voltage in V','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        title('Charge','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        colorbar_plot = colorbar;
                        colorbar_plot.Label.String = replace(variable_for_color,"_"," ");
                        colorbar_plot.Label.Interpreter = 'latex';
                        colorbar_plot.Label.FontSize = FontSizePlots;
                        colorbar_plot.Label.FontName = FontNamePlots;
                        set(gca,'FontSize',FontSizePlots)
                        set(gca,'FontName',FontNamePlots)
                        grid on
                        colormap(mycolor)   
                        caxis(color_lim)
    %                     xlim(ocv_x_lim);
    %                     ylim(ocv_y_lim);
                        set(gca, 'XDir','normal');
                       
    
    
                        file_name = strcat(replace(cell_name," ","_"),"_ocv");
                        save_path = strcat(path_to_data,"figures\",file_name);
    
                        if hide_values
                            axes(subp(2))
    %                         set(gca,'xtick',[])
                            set(gca,'xticklabel',[])
    %                         set(gca,'ytick',[])
                            set(gca,'yticklabel',[])
                            axes(subp(1))
    %                         set(gca,'xtick',[])
                            set(gca,'xticklabel',[])
    %                         set(gca,'ytick',[])
                            set(gca,'yticklabel',[])
                        end
                        if hide_name == 0
                            sgtitle(convertStringsToChars(strcat('\textbf{',title_name,'}')),'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
                        end
    
                        if save_plots
                            drawnow;
    
                            if exist('exportgraphics')
                                set(gcf, 'color', 'none'); 
                                axes(subp(2))
                                set(gca, 'color', 'none');
                                axes(subp(1))
                                set(gca, 'color', 'none');
    %                                 savefig(strcat(save_path,'.fig'));
    %                                 exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %                                 exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %                                 exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                                exportgraphics(gcf,strcat(save_path,'.png'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                            else
    %                                 savefig(strcat(save_path,'.fig'));
    %                                 saveas(gcf,save_path,'svg');
    %                                 saveas(gcf,save_path,'pdf');
    %                                 saveas(gcf,save_path,'emf');
                                saveas(gcf,save_path,'png');
                            end
                        end
                    end
                end
            end
        end
    end

    fprintf('%s x_plot_ocv:\t\t\t %f s\n',cell_name,toc);
end