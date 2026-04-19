function TT = capacity_fade_analysis_function(cell_name,path_to_data,compression,csv_export,T_nom_cap,normalize_capa)
%% start tic to display the total time requried at the end
tic
% %% config arguments
% for i = 1:2:length(varargin)
%     eval(strcat(varargin{i},"=",varargin{i+1},";"))
% end
close all

figure_width_cm = 16;
figure_height_cm = figure_width_cm/3*2;
FontSizePlots = 10;
FontSizeColorbar = 10;
FontSizeTitle = 8;
LineWidthPlots = 1.0;
LineWidthLimitsPlots = 2.0;
FontNamePlots = "arial";
title_name = "";

save_plots = 1;
color_steps_quantisation = 10000;

normalize_pulse = 0;

%% load the timeseries and delete EIS if exist
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.Current(1);
    cell_name_ahjo = split(cell_name,"-PROJECT");
catch
    return

end

if any("Capacity" == string(TT.Properties.VariableNames))

    ind_capa = TT.Capacity>0 & TT.Capacity_current<0;
    capacity = TT.Capacity(ind_capa);
    capacity_current = TT.Capacity_current(ind_capa);
    ah_throughput = TT.Ah_throughput(ind_capa);

    if (sum(ind_capa) >0 && not(isnan(sum(capacity))) && not(isnan(sum(capacity_current)))&& not(isnan(sum(ah_throughput)))) == true

        try
            c_nom = T_nom_cap.("Nominal Capacity in Ah")(T_nom_cap.("Cell Name")==cell_name_ahjo(1));
        catch
            c_nom = max(capacity);
        end

        if isnan(c_nom)
            c_nom = max(capacity);
        end

        if isempty(c_nom)
            c_nom = max(capacity);
        end

        if normalize_capa ==1
            capacity = capacity / c_nom * 100;
            capacity_current = capacity_current / c_nom;
            ah_throughput = ah_throughput / 2/c_nom;
        end


        [N,edges] = histcounts(capacity_current,color_steps_quantisation);
        capacity_current_discret = edges(discretize(capacity_current,edges));

        color_lim = [min(capacity_current_discret), max(capacity_current_discret)];
        color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);

        set(0,'defaulttextinterpreter','latex')
        set(0,'defaultAxesTickLabelInterpreter','latex');
        set(0,'defaultLegendInterpreter','latex');

        mycolor = flip(turbo(color_steps_quantisation));
        fig = figure('Name',strcat(cell_name, "Capacity EFC"),'Visible','Off');
        set(fig,'Units','centimeters')
        set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        hold on


        for cap_value = 1:length(capacity)
            [minValue, colorIndex] = min(abs(color_steps-capacity_current_discret(cap_value)));
            plot(ah_throughput(cap_value), capacity(cap_value),'x', 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        end
        colormap(mycolor)
        colorbar_plot = colorbar;
        colorbar_plot.Label.Interpreter = 'latex';
        if normalize_capa ==1
            colorbar_plot.Label.String = strcat("$\mathrm{Current\ for\ Capacity\ Measurement\ in\ C-Rate}$");
        else
            colorbar_plot.Label.String = strcat("$\mathrm{Current\ for\ Capacity\ Measurement\ in\ A}$");
        end
        colorbar_plot.FontSize = FontSizeTitle;
        colorbar_plot.Label.FontSize = FontSizeColorbar;
        colorbar_plot.Label.FontName = FontNamePlots;
        colorbar_plot.Location = 'northoutside';

        if diff(color_lim) > 0
            caxis(color_lim)
        end

        if normalize_capa ==1
            xlabel('$\mathrm{Equivalent\ Full\ Cycle\ (EFC)}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            ylabel('$\mathrm{Normalised \ Capacity\ (SOH_C)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        else
            xlabel('$\mathrm{Ampere}$-$\mathrm{hour\ Throughput}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            ylabel('$\mathrm{Capacity\ in\ Ah}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        end
        grid on

        if normalize_capa ==1
            plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

            if min(capacity) < 80
                plot(get(gca,'xlim'),[80 80],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            end
        end


        file_name = strcat(replace(cell_name," ","_"));
        save_path = strcat(path_to_data,"figures\",file_name,"_capcity_EFC");
        % drawnow;
        if exist('exportgraphics')
            set(gcf, 'color', 'none');
            set(gca, 'color', 'none');
            %     savefig(strcat(save_path,'.fig'));
            %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
            %     savefig(strcat(save_path,'.fig'));
            %     saveas(gcf,save_path,'svg');
            %     saveas(gcf,save_path,'pdf');
            %     saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end
        close all

        fig = figure('Name',strcat(cell_name, "Capacity Time"),'Visible','Off');
        set(fig,'Units','centimeters')
        set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        hold on

        try
            duration = TT.Duration(ind_capa);
        catch
            TT.Duration = days(TT.Time(:)-TT.Time(1));
            duration = TT.Duration(ind_capa);
        end

        for cap_value = 1:length(capacity)
            [minValue, colorIndex] = min(abs(color_steps-capacity_current_discret(cap_value)));
            plot(duration(cap_value), capacity(cap_value),'x', 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        end
        colormap(mycolor)
        if diff(color_lim) > 0
            colorbar_plot = colorbar;
            colorbar_plot.Label.Interpreter = 'latex';
            if normalize_capa ==1
                colorbar_plot.Label.String = strcat("$\mathrm{Current\ for\ Capacity\ Measurement\ in\ C-Rate}$");
            else
                colorbar_plot.Label.String = strcat("$\mathrm{Current\ for\ Capacity\ Measurement\ in\ A}$");
            end
            colorbar_plot.FontSize = FontSizeTitle;
            colorbar_plot.Label.FontSize = FontSizeColorbar;
            colorbar_plot.Label.FontName = FontNamePlots;
            colorbar_plot.Location = 'northoutside';
            caxis(color_lim)
        end

        xlabel('$\mathrm{Time\ in\ days}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        if normalize_capa ==1
            ylabel('$\mathrm{Normalised \ Capacity\ (SOH_C)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        else
            ylabel('$\mathrm{Capacity\ in\ Ah}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        end
        grid on

        if normalize_capa ==1
            plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

            if min(capacity) < 80
                plot(get(gca,'xlim'),[80 80],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            end
        end


        file_name = strcat(replace(cell_name," ","_"));
        save_path = strcat(path_to_data,"figures\",file_name,"_capcity_time");
        % drawnow;
        if exist('exportgraphics')
            set(gcf, 'color', 'none');
            set(gca, 'color', 'none');
            %     savefig(strcat(save_path,'.fig'));
            %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
            %     savefig(strcat(save_path,'.fig'));
            %     saveas(gcf,save_path,'svg');
            %     saveas(gcf,save_path,'pdf');
            %     saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end
    end
end


%% pulse evaulation
if any("Pulse_Resistance_Duration" == string(TT.Properties.VariableNames)) && any("Capacity" == string(TT.Properties.VariableNames))

    try
        c_nom = T_nom_cap.("Nominal Capacity in Ah")(T_nom_cap.("Cell Name")==cell_name_ahjo(1));
    catch
        c_nom = max(capacity);
    end

    if isempty(c_nom)
        c_nom = max(capacity);
    end

    Pulse_Resistance_Duration = TT.Pulse_Resistance_Duration(not(isnan(TT.Pulse_Resistance_Duration)));
    Pulse_Resistance =  TT.Pulse_Resistance(not(isnan(TT.Pulse_Resistance_Duration)));
    Pulse_Resistance_Area = TT.Pulse_Resistance_Area(not(isnan(TT.Pulse_Resistance_Duration)));
    Pulse_Resistance_Current = TT.Pulse_Resistance_Current(not(isnan(TT.Pulse_Resistance_Duration)));

    if normalize_pulse ==1
        Pulse_Resistance = (Pulse_Resistance-min(Pulse_Resistance))/min(Pulse_Resistance)*100;
        Pulse_Resistance_Area = (Pulse_Resistance_Area-min(Pulse_Resistance_Area))/min(Pulse_Resistance_Area)*100;
    end

    max_pulse_length =120;
    delta_groups = 1;
    pulse_duration_discret = discretize(Pulse_Resistance_Duration,linspace(0+delta_groups/2,max_pulse_length+delta_groups/2,max_pulse_length/delta_groups+1));

    symbols_groups = unique(pulse_duration_discret);
    symbols_groups(isnan(symbols_groups)) = [];
    required_symbols = length(unique(pulse_duration_discret));

    


    symbols = ["o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram","o","+","*",".","x","_","|","square","diamond","^","v",">","<","pentagram","hexagram"];

    %symbols(symbols_groups == pulse_duration_discret(1))


    [N,edges] = histcounts(Pulse_Resistance_Current,color_steps_quantisation);
    Pulse_Resistance_Current_discret = edges(discretize(Pulse_Resistance_Current,edges));

    ind_pulses = not(isnan(TT.Pulse_Resistance_Current));

    if required_symbols
        ind_pulses = [];
    end

    if sum(ind_pulses) >0 && not(isempty(Pulse_Resistance_Current_discret))

        TT = fillmissing(TT,'previous','DataVariables',['Capacity']);
        capacity_pulse = TT.Capacity;



        capacity_pulse = capacity_pulse(ind_pulses);

        if normalize_capa ==1
            capacity_pulse = capacity_pulse / c_nom * 100;
        end



        ah_throughput = TT.Ah_throughput(ind_pulses);

        if normalize_capa ==1
            ah_throughput = ah_throughput / 2/c_nom;
        end

        duration = TT.Duration(ind_pulses);
        close all
        fig = figure('Name',strcat(cell_name, "Pulse EFC"),'Visible','Off');
        set(fig,'Units','centimeters')
        set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        hold on

        color_lim = [min(Pulse_Resistance_Current_discret), max(Pulse_Resistance_Current_discret)];
        color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);
        mycolor = flip(turbo(color_steps_quantisation));




        symbols_table = strings(length(ah_throughput),1);
        colors_table = zeros(length(ah_throughput),3);
        for ind_pulse = 1:length(Pulse_Resistance_Duration)
            if isnan(pulse_duration_discret(ind_pulse))
                continue
            end
            [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
            symbols_table(ind_pulse) = symbols(symbols_groups == pulse_duration_discret(ind_pulse));
            colors_table(ind_pulse,:) = mycolor(colorIndex, :);
        end
        scatter_table = table(ah_throughput,Pulse_Resistance,symbols_table,colors_table);
        scatter_table = sortrows(scatter_table,'symbols_table','ascend');
        unique_symbols = unique(scatter_table.symbols_table);
        for symbol_ind = 1:length(unique_symbols)
            sup_table = scatter_table(scatter_table.symbols_table == unique_symbols(symbol_ind),:);
            scatter(sup_table.ah_throughput,sup_table.Pulse_Resistance*1000,[],sup_table.colors_table,"filled",unique_symbols(symbol_ind));
        end

        %         for ind_pulse = 1:length(Pulse_Resistance_Duration)
        %             [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
        %             plot(ah_throughput(ind_pulse), Pulse_Resistance(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        %         end

        colormap(mycolor)
        if diff(color_lim) > 0
            colorbar_plot = colorbar;
            colorbar_plot.Label.Interpreter = 'latex';
            colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
            colorbar_plot.FontSize = FontSizeTitle;
            colorbar_plot.Label.FontSize = FontSizeColorbar;
            colorbar_plot.Label.FontName = FontNamePlots;
            colorbar_plot.Location = 'northoutside';
            caxis(color_lim)
        end


        if normalize_pulse ==1
            ylabel('$\mathrm{Normalised \ Resistance increase\ (SOH_R)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        else
            ylabel('$\mathrm{Resistance\ in\ m\Omega}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        end

        if normalize_capa ==1
            xlabel('$\mathrm{Equivalent\ Full\ Cycle\ (EFC)}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        else
            xlabel('$\mathrm{Ampere}$-$\mathrm{hour\ Throughput}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        end


        grid on

        if normalize_pulse ==1
            plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

            if max(Pulse_Resistance) >= 200
                plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            end
        end

        hold on;

        h = zeros(required_symbols, 1);
        for i = 1:required_symbols
            h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
        end

        legend_name = string(num2str(symbols_groups,"%d"));
        legend_name = strcat(legend_name," s");
        legend(h, legend_name, 'location', 'best');


        file_name = strcat(replace(cell_name," ","_"));
        save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_EFC");
        % drawnow;
        if exist('exportgraphics')
            set(gcf, 'color', 'none');
            set(gca, 'color', 'none');
            %     savefig(strcat(save_path,'.fig'));
            %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
            %     savefig(strcat(save_path,'.fig'));
            %     saveas(gcf,save_path,'svg');
            %     saveas(gcf,save_path,'pdf');
            %     saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end

        close all
        %         fig = figure('Name',strcat(cell_name, "Pulse Area EFC"),'Visible','Off');
        %         fig.Name = strcat(cell_name, "Pulse Area EFC");
        %         set(fig,'Units','centimeters')
        %         set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        %         hold on
        %
        %
        %         for ind_pulse = 1:length(Pulse_Resistance_Duration)
        %             [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
        %             plot(ah_throughput(ind_pulse), Pulse_Resistance_Area(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        %         end
        %
        %         colormap(mycolor)
        %         if diff(color_lim) > 0
        %             colorbar_plot = colorbar;
        %             colorbar_plot.Label.Interpreter = 'latex';
        %             colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
        %             colorbar_plot.FontSize = FontSizeTitle;
        %             colorbar_plot.Label.FontSize = FontSizeColorbar;
        %             colorbar_plot.Label.FontName = FontNamePlots;
        %             colorbar_plot.Location = 'northoutside';
        %              caxis(color_lim)
        %         end
        %
        %         xlabel('$\mathrm{Equivalent\ Full\ Cycle\ (EFC)}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         if normalize_pulse ==1
        %             ylabel('$\mathrm{Normalised\ Area\ under\ Pulse\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         else
        %             ylabel('$\mathrm{Resistance\ in\ \frac{Vs}{Is}}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         end
        %         grid on
        %
        %         if normalize_pulse ==1
        %             plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
        %
        %             if max(Pulse_Resistance_Area) >= 200
        %                 plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
        %             end
        %         end
        %
        %         hold on;
        %
        %         h = zeros(required_symbols, 1);
        %         for i = 1:required_symbols
        %             h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
        %         end
        %
        %         legend_name = string(num2str(symbols_groups,"%d"));
        %         legend_name = strcat(legend_name," s");
        %         legend(h, legend_name, 'location', 'best');
        %
        %
        %         file_name = strcat(replace(cell_name," ","_"));
        %         save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_area_EFC");
        %         % drawnow;
        %         if exist('exportgraphics')
        %             set(gcf, 'color', 'none');
        %             set(gca, 'color', 'none');
        %         %     savefig(strcat(save_path,'.fig'));
        %         %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %         %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %         %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %             exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        %         else
        %         %     savefig(strcat(save_path,'.fig'));
        %         %     saveas(gcf,save_path,'svg');
        %         %     saveas(gcf,save_path,'pdf');
        %         %     saveas(gcf,save_path,'emf');
        %             saveas(gcf,save_path,'png');
        %         end

        close all
        fig = figure('Name',strcat(cell_name, "Pulse time"),'Visible','Off');
        set(fig,'Units','centimeters')
        set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        hold on

        color_lim = [min(Pulse_Resistance_Current_discret), max(Pulse_Resistance_Current_discret)];
        color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);


        symbols_table = strings(length(ah_throughput),1);
        colors_table = zeros(length(ah_throughput),3);
        for ind_pulse = 1:length(Pulse_Resistance_Duration)
            if isnan(pulse_duration_discret(ind_pulse))
                continue
            end
            [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
            symbols_table(ind_pulse) = symbols(symbols_groups == pulse_duration_discret(ind_pulse));
            colors_table(ind_pulse,:) = mycolor(colorIndex, :);
        end
        scatter_table = table(duration,Pulse_Resistance,symbols_table,colors_table);
        scatter_table = sortrows(scatter_table,'symbols_table','ascend');
        unique_symbols = unique(scatter_table.symbols_table);
        for symbol_ind = 1:length(unique_symbols)
            sup_table = scatter_table(scatter_table.symbols_table == unique_symbols(symbol_ind),:);
            scatter(sup_table.duration,sup_table.Pulse_Resistance*1000,[],sup_table.colors_table,"filled",unique_symbols(symbol_ind));
        end


        %         for ind_pulse = 1:length(Pulse_Resistance_Duration)
        %             [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
        %             plot(duration(ind_pulse), Pulse_Resistance(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        %         end

        colormap(mycolor)
        if diff(color_lim) > 0
            colorbar_plot = colorbar;
            colorbar_plot.Label.Interpreter = 'latex';
            colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
            colorbar_plot.FontSize = FontSizeTitle;
            colorbar_plot.Label.FontSize = FontSizeColorbar;
            colorbar_plot.Label.FontName = FontNamePlots;
            colorbar_plot.Location = 'northoutside';
            caxis(color_lim)
        end

        xlabel('$\mathrm{Time\ in\ days}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        if normalize_pulse ==1
            ylabel('$\mathrm{Normalised \ Resistance increase\ (SOH_R)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        else
            ylabel('$\mathrm{Resistance\ in\ m\Omega}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        end
        grid on

        if normalize_pulse ==1
            plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

            if max(Pulse_Resistance) >= 200
                plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            end
        end

        hold on;

        h = zeros(required_symbols, 1);
        for i = 1:required_symbols
            h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
        end

        legend_name = string(num2str(symbols_groups,"%d"));
        legend_name = strcat(legend_name," s");
        legend(h, legend_name, 'location', 'best');


        file_name = strcat(replace(cell_name," ","_"));
        save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_time");
        % drawnow;
        if exist('exportgraphics')
            set(gcf, 'color', 'none');
            set(gca, 'color', 'none');
            %     savefig(strcat(save_path,'.fig'));
            %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
            %     savefig(strcat(save_path,'.fig'));
            %     saveas(gcf,save_path,'svg');
            %     saveas(gcf,save_path,'pdf');
            %     saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end
        close all

        %         fig = figure('Name',strcat(cell_name, "Pulse Area Time"),'Visible','Off');
        %         set(fig,'Units','centimeters')
        %         set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
        %         hold on
        %
        %
        %         for ind_pulse = 1:length(Pulse_Resistance_Duration)
        %             [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
        %             plot(duration(ind_pulse), Pulse_Resistance_Area(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
        %         end
        %
        %         colormap(mycolor)
        %         if diff(color_lim) > 0
        %             colorbar_plot = colorbar;
        %             colorbar_plot.Label.Interpreter = 'latex';
        %             colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
        %             colorbar_plot.FontSize = FontSizeTitle;
        %             colorbar_plot.Label.FontSize = FontSizeColorbar;
        %             colorbar_plot.Label.FontName = FontNamePlots;
        %             colorbar_plot.Location = 'northoutside';
        %              caxis(color_lim)
        %         end
        %
        %         xlabel('$\mathrm{Time\ in\ days}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         if normalize_pulse ==1
        %             ylabel('$\mathrm{Normalised\ Area\ under\ Pulse\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         else
        %             ylabel('$\mathrm{Resistance\ in\ \frac{Vs}{Is}}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
        %         end
        %         grid on
        %
        %         if normalize_pulse ==1
        %             plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
        %
        %             if max(Pulse_Resistance_Area) >= 200
        %                 plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
        %             end
        %         end
        %
        %         hold on;
        %
        %         h = zeros(required_symbols, 1);
        %         for i = 1:required_symbols
        %             h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
        %         end
        %
        %         legend_name = string(num2str(symbols_groups,"%d"));
        %         legend_name = strcat(legend_name," s");
        %         legend(h, legend_name, 'location', 'best');
        %
        %
        %         file_name = strcat(replace(cell_name," ","_"));
        %         save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_area_time");
        %         % drawnow;
        %         if exist('exportgraphics')
        %             set(gcf, 'color', 'none');
        %             set(gca, 'color', 'none');
        %         %     savefig(strcat(save_path,'.fig'));
        %         %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %         %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %         %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
        %             exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        %         else
        %         %     savefig(strcat(save_path,'.fig'));
        %         %     saveas(gcf,save_path,'svg');
        %         %     saveas(gcf,save_path,'pdf');
        %         %     saveas(gcf,save_path,'emf');
        %             saveas(gcf,save_path,'png');
        %         end


        %% pulse and capacity

        capacity_pulse = TT.Capacity;
        capacity_pulse = capacity_pulse(not(isnan(TT.Pulse_Resistance_Duration)));
        if normalize_capa ==1
            capacity_pulse = capacity_pulse / c_nom * 100;
        end

        if sum(capacity_pulse) > 0
            close all
            fig = figure('Name',strcat(cell_name, "Pulse capacity"),'Visible','Off');
            set(fig,'Units','centimeters')
            set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
            hold on

            color_lim = [min(Pulse_Resistance_Current_discret), max(Pulse_Resistance_Current_discret)];
            color_steps = linspace(color_lim(1), color_lim(2),color_steps_quantisation);


            symbols_table = strings(length(ah_throughput),1);
            colors_table = zeros(length(ah_throughput),3);
            for ind_pulse = 1:length(Pulse_Resistance_Duration)
                if isnan(pulse_duration_discret(ind_pulse))
                    continue
                end
                [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
                symbols_table(ind_pulse) = symbols(symbols_groups == pulse_duration_discret(ind_pulse));
                colors_table(ind_pulse,:) = mycolor(colorIndex, :);
            end
            scatter_table = table(capacity_pulse,Pulse_Resistance,symbols_table,colors_table);
            scatter_table = sortrows(scatter_table,'symbols_table','ascend');
            unique_symbols = unique(scatter_table.symbols_table);
            for symbol_ind = 1:length(unique_symbols)
                sup_table = scatter_table(scatter_table.symbols_table == unique_symbols(symbol_ind),:);
                scatter(sup_table.capacity_pulse,sup_table.Pulse_Resistance*1000,[],sup_table.colors_table,"filled",unique_symbols(symbol_ind));
            end


            %             for ind_pulse = 1:length(Pulse_Resistance_Duration)
            %                 [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
            %                 plot(capacity_pulse(ind_pulse), Pulse_Resistance(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
            %             end



            set ( gca, 'xdir', 'reverse' )

            colormap(mycolor)
            if diff(color_lim) > 0
                colorbar_plot = colorbar;
                colorbar_plot.Label.Interpreter = 'latex';
                colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
                colorbar_plot.FontSize = FontSizeTitle;
                colorbar_plot.Label.FontSize = FontSizeColorbar;
                colorbar_plot.Label.FontName = FontNamePlots;
                colorbar_plot.Location = 'northoutside';
                caxis(color_lim)
            end

            if normalize_capa ==1
                xlabel('$\mathrm{Normalised \ Capacity\ (SOH_C)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            else
                xlabel('$\mathrm{Capacity\ in\ Ah}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            end
            if normalize_pulse ==1
                ylabel('$\mathrm{Normalised \ Resistance increase\ (SOH_R)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            else
                ylabel('$\mathrm{Resistance\ in\ m\Omega}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            end
            grid on

            if normalize_pulse ==1
                plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

                if max(Pulse_Resistance) >= 200
                    plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
                end
            end

            if normalize_capa ==1
                plot([100 100 ],get(gca,'ylim'),':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)

                if min(capacity_pulse) < 80
                    plot([80 80],get(gca,'ylim'),':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
                end
            end

            hold on;

            h = zeros(required_symbols, 1);
            for i = 1:required_symbols
                h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
            end

            legend_name = string(num2str(symbols_groups,"%d"));
            legend_name = strcat(legend_name," s");
            legend(h, legend_name, 'location', 'best');


            file_name = strcat(replace(cell_name," ","_"));
            save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_capacity");
            % drawnow;
            if exist('exportgraphics')
                set(gcf, 'color', 'none');
                set(gca, 'color', 'none');
                %     savefig(strcat(save_path,'.fig'));
                %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
                exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
            else
                %     savefig(strcat(save_path,'.fig'));
                %     saveas(gcf,save_path,'svg');
                %     saveas(gcf,save_path,'pdf');
                %     saveas(gcf,save_path,'emf');
                saveas(gcf,save_path,'png');
            end

            close all
            %         fig = figure('Name',strcat(cell_name, "Pulse Area Capacity"),'Visible','Off');
            %         set(fig,'Units','centimeters')
            %         set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
            %         hold on
            %
            %
            %         for ind_pulse = 1:length(Pulse_Resistance_Duration)
            %             [minValue, colorIndex] = min(abs(color_steps-Pulse_Resistance_Current_discret(ind_pulse)));
            %             plot(capacity_pulse(ind_pulse), Pulse_Resistance_Area(ind_pulse),symbols(symbols_groups == pulse_duration_discret(ind_pulse)), 'color', mycolor(colorIndex, :),'LineWidth',LineWidthPlots);
            %         end
            %
            %         set ( gca, 'xdir', 'reverse' )
            %
            %         colormap(mycolor)
            %         if diff(color_lim) > 0
            %             colorbar_plot = colorbar;
            %             colorbar_plot.Label.Interpreter = 'latex';
            %             colorbar_plot.Label.String = strcat("$\mathrm{Current\ of\ the\ Pulse\ in\ C}$");
            %             colorbar_plot.FontSize = FontSizeTitle;
            %             colorbar_plot.Label.FontSize = FontSizeColorbar;
            %             colorbar_plot.Label.FontName = FontNamePlots;
            %             colorbar_plot.Location = 'northoutside';
            %              caxis(color_lim)
            %         end
            %
            %         if normalize_capa ==1
            %             xlabel('$\mathrm{Normalised \ Capacity\ (SOH_C)\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            %         else
            %             xlabel('$\mathrm{Capacity\ in\ Ah}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            %         end
            %         if normalize_pulse ==1
            %             ylabel('$\mathrm{Normalised\ Area\ under\ Pulse\ in\ \%}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            %         else
            %             ylabel('$\mathrm{Resistance\ in\ \frac{Vs}{Is}}$','Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
            %         end
            %         grid on
            %
            %         if normalize_pulse ==1
            %             plot(get(gca,'xlim'),[100 100 ],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            %
            %             if max(Pulse_Resistance_Area) >= 200
            %                 plot(get(gca,'xlim'),[200 200],':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            %             end
            %         end
            %
            %         if normalize_capa ==1
            %             plot([100 100 ],get(gca,'ylim'),':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            %
            %             if min(capacity_pulse) < 80
            %                 plot([80 80],get(gca,'ylim'),':','Color',[156/255,158/255,159/255],'LineWidth',LineWidthLimitsPlots)
            %             end
            %         end
            %
            %         hold on;
            %
            %         h = zeros(required_symbols, 1);
            %         for i = 1:required_symbols
            %             h(i) = plot(NaN,NaN,symbols(i),'MarkerEdgeColor',"black");
            %         end
            %
            %         legend_name = string(num2str(symbols_groups,"%d"));
            %         legend_name = strcat(legend_name," s");
            %         legend(h, legend_name, 'location', 'best');
            %
            %
            %         file_name = strcat(replace(cell_name," ","_"));
            %         save_path = strcat(path_to_data,"figures\",file_name,"_pulse_resistance_area_capacity");
            %         % drawnow;
            %         if exist('exportgraphics')
            %             set(gcf, 'color', 'none');
            %             set(gca, 'color', 'none');
            %         %     savefig(strcat(save_path,'.fig'));
            %         %     exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %         %     exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %         %     exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            %             exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
            %         else
            %         %     savefig(strcat(save_path,'.fig'));
            %         %     saveas(gcf,save_path,'svg');
            %         %     saveas(gcf,save_path,'pdf');
            %         %     saveas(gcf,save_path,'emf');
            %             saveas(gcf,save_path,'png');
            %         end
            %     end
        end
    end

    %% finished, display the required time for the complete cell
    fprintf('%s capacity_fade_analysis_function:\t\t\t %f s\n',cell_name,toc);
end
