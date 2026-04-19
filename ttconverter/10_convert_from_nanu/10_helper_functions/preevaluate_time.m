function [] = preevaluate_time(varargin)
%% start tic to display the total time requried at the end
tic

close all


cell_name = "LBE_Samsung_35E_137";
path_to_data = "./../../../data/";
save_plots = 1;
time_plot = 1;
statistics = 1;
hide_name =1;

figure_width_cm = 44;
figure_height_cm = 25;
FontSizePlots = 10;
FontSizePlotsLabel = 8;
FontNamePlots = "arial";

nan_threshold_s = 125;

table_size_limit = 10000000;


%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=",varargin{i+1},";"))
end


title_name = replace(cell_name,"_"," ");


% load the EIS timeseries 
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.Voltage(1); % test if data is accessible
catch
    warning(strcat("Timeseries file could not be load: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    TT.EIS_Frequency = [];
    TT.EIS_Z_abs = [];
    TT.EIS_Z_phase = [];
end

indexNaN = find(seconds(diff(TT.Time)) > nan_threshold_s);

TT{indexNaN,:} = missing;

TT(TT.Time < datetime("1980-01-01 00:00:00",'TimeZone','Europe/Berlin'),:) = [];

TT{TT.Temperature < -100,"Temperature"} = NaN;
TT{TT.Temperature > 100,"Temperature"} = NaN;

TT{TT.Voltage > 5,"Voltage"} = NaN;
TT{TT.Voltage < 0.5,"Voltage"} = NaN;



% detail level
detail_level = 1;
% 1:basic           Voltage, Current, Temperature 
% 2:with AH         Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput
% 3:with AH, WH     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput
% 4:with AH and Cap Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput, Capacity, Capacity_current
% 5:AH, WH, Cap     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput, Capacity, Capacity_current
% 5:AH, WH, Cap     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput, Capacity, Capacity_current
try
    TT.Capacity(1);
    try
        TT.Wh_Counter(1);
        detail_level = 5;
    catch
        detail_level = 4;
    end
catch
    try
        TT.Wh_Counter(1);
        detail_level = 3;
    catch
            try
                TT.Ah_throughput(1);
                detail_level = 2;
            catch
                detail_level = 1;
            end
    end
end


if size(TT,1) > table_size_limit
    if detail_level >3
        TT_slow = TT(:,{'Capacity', 'Capacity_current'});
        % remove data with missing values
        TF  = ismissing(TT_slow.Capacity);
        TT_slow = TT_slow(~TF,:);
        clear TF

        TT = removevars(TT,{'Capacity', 'Capacity_current'});
        TT = TT(round(linspace(1,size(TT,1),table_size_limit)),:);

        TT = synchronize(TT, TT_slow);

    else
        TT = TT(round(linspace(1,size(TT,1),table_size_limit)),:);
    end
end


if time_plot
    fig = figure();
    set(fig,'Units','centimeters')
    set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
    set(gca,'FontSize',FontSizePlots)
    set(gca,'FontName',FontNamePlots)

    s = stackedplot(TT,{'Voltage','Current','Temperature'});
    
    
    
    s.DisplayLabels = {{'Voltage in V'},{'Current in A'},{'Temperature in $^\circ$C'}};

    ax = findobj(s.NodeChildren, 'Type','Axes');

    % ax(3).YScale = 'log';
    if hide_name == 0
        sgtitle(convertStringsToChars(strcat('\textbf{',title_name,'}')),'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
    end
    
    xlabel('Date of Measurement')
    drawnow;
    
    for i = 1:3
        ax(i,1);
        ax(i,1).XGrid = 'on';
        ax(i,1).YGrid = 'on';
        set(ax(i,1).XLabel,'Interpreter','latex');
        set([ax(i,1).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','Interpreter','latex');
        set(ax(i,1),'FontSize',FontSizePlots)
        set(ax(i,1),'FontName',FontNamePlots)
        ax(i,1).TickLabelInterpreter = 'latex';

        if save_plots
            if exist('exportgraphics')
                set(ax(i,1), 'color', 'none');
            end
        end
    end
    
    
    
    grid on
    
    
    if save_plots
        
        file_name = strcat(replace(cell_name," ","_"),...
        "_pre-evaluation_time");
        save_path = strcat(path_to_data,"figures\",file_name);

        if exist('exportgraphics')
    %         savefig(strcat(save_path,'.fig'));
    %         exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %         exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %         exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
    %         savefig(strcat(save_path,'.fig'));
    %         saveas(gcf,save_path,'svg');
    %         saveas(gcf,save_path,'pdf');
    %         saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end
        close all
        clear s
        clear ax
    end
end



%% statistics
if statistics

    fig = figure();
    set(fig,'Units','centimeters')
    set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
    % [S,AX,BigAx,H,HAx] = plotmatrix([voltages,temperatures,currents,ah_througputs,durations]);

% 1:basic           Voltage, Current, Temperature 
% 2:with AH         Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput
% 3:with AH, WH     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput
% 4:with AH and Cap Voltage, Current, Temperature, Duration, Ah_counter, Ah_throughput, Capacity, Capacity_current
% 5:AH, WH, Cap     Voltage, Current, Temperature, Duration, Ah_counter, Wh_counter, Ah_throughput, Wh_throughput, Capacity, Capacity_current

    switch detail_level
        case 1
            [h,ax,bigax] = gplotmatrix([TT.Voltage,TT.Current, TT.Temperature],[],[],[], [],[],[],...
                'hist',{'Voltage in V','Current in A','Temperature in $^\circ$C'});
        case 2
            [h,ax,bigax] = gplotmatrix([TT.Voltage,TT.Current, TT.Temperature, TT.Duration, TT.Ah_throughput],[],[],[], [],[],[],...
                'hist',{'Voltage in V','Current in A','Temperature in $^\circ$C','Duration in days','Ah Throughput in Ah'});
        case 3
            [h,ax,bigax] = gplotmatrix([TT.Voltage,TT.Current, TT.Temperature, TT.Duration, TT.Ah_throughput, TT.Wh_throughput],[],[],[], [],[],[],...
                'hist',{'Voltage in V','Current in A','Temperature in $^\circ$C','Duration in days','Ah Throughput in Ah','Wh Throughput in Wh'});
        case 4
            [h,ax,bigax] = gplotmatrix([TT.Voltage,TT.Current, TT.Temperature, TT.Duration, TT.Ah_throughput, TT.Capacity, TT.Capacity_current],[],[],[], [],[],[],...
            'hist',{'Voltage in V','Current in A','Temperature in $^\circ$C','Duration in days','Ah Throughput in Ah','Capacity in Ah','Cur. of cap.-test in A'});
            % flip capacity x-axis
            for y = 1:size(ax,1)
                set ( ax(y,6), 'xdir', 'reverse' )
            end
        case 5
            [h,ax,bigax] = gplotmatrix([TT.Voltage,TT.Current, TT.Temperature, TT.Duration, TT.Ah_throughput, TT.Wh_throughput, TT.Capacity, TT.Capacity_current],[],[],[], [],[],[],...
            'hist',{'Voltage in V','Current in A','Temperature in $^\circ$C','Duration in days','Ah Throughput in Ah','Wh Throughput in Wh','Capacity in Ah','Cur. of cap.-test in A'});
            % flip capacity x-axis
            for y = 1:size(ax,1)
                set ( ax(y,6), 'xdir', 'reverse' )
            end
    end

    clear TT
    
    
    
    for y = 1:size(ax,1)
        for x = 1:size(ax,2)
            ax(y,x).XGrid = 'on';
            ax(y,x).YGrid = 'on';
            set(ax(y,x).XLabel,'Interpreter','latex');
            set(ax(y,x).YLabel,'Interpreter','latex');
            set(ax(y,x),'FontSize',FontSizePlotsLabel)
            set(ax(y,x),'FontName',FontNamePlots)
            
            
            ax(y,x).TickLabelInterpreter = 'latex';
    
            if save_plots
                if exist('exportgraphics')
                    axes(ax(y,x));
                    set(gca, 'color', 'none');
                end
            end
        end
    end
    
    if hide_name == 0
        sgtitle(convertStringsToChars(strcat('\textbf{',title_name,'}')),'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);
    end
    
    if save_plots
        
        file_name = strcat(replace(cell_name," ","_"),...
        "_time_stat");
        save_path = strcat(path_to_data,"figures\",file_name);
        
        
        drawnow;
        if exist('exportgraphics')
    %         savefig(strcat(save_path,'.fig'));
    %         exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %         exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
    %         exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none','Resolution',300);
            exportgraphics(gcf,strcat(save_path,'.png'),'Resolution',300);
        else
    %         savefig(strcat(save_path,'.fig'));
    %         saveas(gcf,save_path,'svg');
    %         saveas(gcf,save_path,'pdf');
    %         saveas(gcf,save_path,'emf');
            saveas(gcf,save_path,'png');
        end
        close all
    end
end


%% finished, display the required time for the complete cell
fprintf('%s preevaluate_time:\t\t\t %f s\n',cell_name,toc);
end