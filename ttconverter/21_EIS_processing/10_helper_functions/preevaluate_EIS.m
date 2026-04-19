function [] = preevaluate_EIS(varargin)

close all

cell_name = "Everlast_35E_033";
path_to_data = "E:\data\";
save_plots = 1;


figure_width_cm = 31.8;
figure_height_cm = 31;
FontSizePlots = 10;
FontNamePlots = "arial";



%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=""",varargin{i+1},""";"))
end


title_name = replace(cell_name,"_"," ");
% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"figures/"));
mkdir(strcat(path_to_data,"eis_data/"));

% load the EIS timeseries 
try
    load(strcat(path_to_data,'eis_data\',cell_name,'_eis.mat'),'TT_eis');
    TT_eis.EIS_Frequency(1); % test if data is accessible

    

catch
    warning(strcat("EIS timeseries file could not be load: ", strcat(path_to_data,'eis_data\',cell_name,'_eis.mat') ));
    return;
end

try
    TT_eis.Capacity(1);
catch
    warning(strcat("EIS does not contain Capacity values: ", strcat(path_to_data,'eis_data\',cell_name,'_eis.mat') ));
    return;
end

measurements = unique(TT_eis.EIS_measurement_id);

voltages = zeros(length(measurements),1);
temperatures = zeros(length(measurements),1);
currents = zeros(length(measurements),1);
ah_througputs = zeros(length(measurements),1);
durations = zeros(length(measurements),1);
duration_min = min(TT_eis.Time);
capacity = zeros(length(measurements),1);
capacity_current = zeros(length(measurements),1);

for id = 1:length(measurements)
    voltages(id) = mean(TT_eis.Voltage(TT_eis.EIS_measurement_id == measurements(id)));
    temperatures(id) = mean(TT_eis.Temperature(TT_eis.EIS_measurement_id == measurements(id)));
    currents(id) = mean(TT_eis.Current(TT_eis.EIS_measurement_id == measurements(id)));
    ah_througputs(id) = mean(TT_eis.Ah_throughput(TT_eis.EIS_measurement_id == measurements(id)));
    durations(id) = days(mean(TT_eis.Time(TT_eis.EIS_measurement_id == measurements(id))) - duration_min);
    capacity(id) = min(TT_eis.Capacity(TT_eis.EIS_measurement_id == measurements(id)));
    capacity_current(id) = max(TT_eis.Capacity_current(TT_eis.EIS_measurement_id == measurements(id)));
end


fig = figure();
set(fig,'Units','centimeters')
set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
% [S,AX,BigAx,H,HAx] = plotmatrix([voltages,temperatures,currents,ah_througputs,durations]);
[h,ax,bigax] = gplotmatrix([voltages,temperatures,currents,ah_througputs,durations, capacity, capacity_current],[],[],[], [],[],[],'hist',{'Voltage in V','Temperature in $^\circ$C','Current in A','Ah throughput in Ah','Duration in days', 'Capacity in Ah','Cur. of cap.-test in A'});

% flip capacity x-axis
for y = 1:size(ax,1)
    set ( ax(y,6), 'xdir', 'reverse' )
end

% gplotmatrix(TT{:,cellstr(string(TT.Properties.VariableNames(:)).')},[],[],[], [],[],[],'hist',cellstr(string(TT.Properties.VariableNames(:)).'));



for y = 1:size(ax,1)
    for x = 1:size(ax,2)
        ax(y,x).XGrid = 'on';
        ax(y,x).YGrid = 'on';
        set(ax(y,x).XLabel,'Interpreter','latex');
        set(ax(y,x).YLabel,'Interpreter','latex');
        set(ax(y,x),'FontSize',FontSizePlots)
        set(ax(y,x),'FontName',FontNamePlots)
        
        
        ax(y,x).TickLabelInterpreter = 'latex';

        if save_plots
            if exist('exportgraphics')
                axes(ax(y,x));
                set(gca, 'color', 'none');
            end
        end
        
%         if x~=y && y < size(ax,1)
%             data = ax(y,x).Children;
%             data_x = data.XData;
%             data_y = data.YData;
%             data = [data_x.', data_y.'];
%             
%             prob = histcounts(TT_eis.Temperature,1000,'Normalization','probability');
%             
%             cluster = 50;
% 
%             opts = statset('Display','off');
%             klist=2:cluster;%the number of clusters you want to try
%             myfunc = @(X,K)(kmeans(X, K));
%             eva = evalclusters(data,myfunc,'CalinskiHarabasz','klist',klist);
%             [idx,C]=kmeans(data,eva.OptimalK,'Distance','cityblock','Replicates',10,'Options',opts);%'Distance','correlation','Replicates',2,
%             
%             
%             axes(ax(y,x));
%             hold on;
%             for i = 1:cluster
%                 plot(data(idx==i,1),data(idx==i,2),'.')
%                 plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3) 
%             end
%             
%         end       
    end
end

sgtitle(convertStringsToChars(strcat('\textbf{',title_name,'}')),'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);

if save_plots
    
    file_name = strcat(replace(cell_name," ","_"),...
    "_eis_pre-evaluation");
    save_path = strcat(path_to_data,"figures\",file_name);
    
    
    drawnow;
    if exist('exportgraphics')
%         savefig(strcat(save_path,'.fig'));
%         exportgraphics(gcf,strcat(save_path,'.svg'),'ContentType','vector','BackgroundColor','none');
%         exportgraphics(gcf,strcat(save_path,'.emf'),'ContentType','vector','BackgroundColor','none');
%         exportgraphics(gcf,strcat(save_path,'.pdf'),'ContentType','vector','BackgroundColor','none');
        exportgraphics(gcf,strcat(save_path,'.png'),'ContentType','vector','BackgroundColor','none','Resolution',600);
    else
%         savefig(strcat(save_path,'.fig'));
%         saveas(gcf,save_path,'svg');
%         saveas(gcf,save_path,'pdf');
%         saveas(gcf,save_path,'emf');
        saveas(gcf,save_path,'png');
    end
end


% figure(2)
% scatterhist(voltages,temperatures,'Kernel','overlay','Direction','out','Location','SouthWest');%'NBins',[round(0.5*length(unique(voltages))),round(0.5*length(unique(temperatures)))]
% 
% 
% 
% s = scatterhistogram(TT_eis,'Voltage','Temperature','NumBins',[length(unique(voltages));length(unique(temperatures))],'HistogramDisplayStyle','smooth','MarkerStyle','.','MarkerSize',50,'MarkerFilled','off','LegendVisible','on');
% 
% s = scatterhistogram(TT_eis,'Voltage','Current','NumBins',[length(unique(TT_eis.Voltage));length(unique(TT_eis.Current))],'HistogramDisplayStyle','smooth','MarkerStyle','.','MarkerSize',50,'MarkerFilled','off','LegendVisible','on');
% s = scatterhistogram(TT_eis,'Voltage','Temperature','NumBins',[length(unique(TT_eis.Voltage));length(unique(TT_eis.Temperature))],'HistogramDisplayStyle','smooth','MarkerStyle','.','MarkerSize',50,'MarkerFilled','off','LegendVisible','on');
% s = scatterhistogram(TT_eis,'Current','Temperature','NumBins',[length(unique(TT_eis.Current));length(unique(TT_eis.Temperature))],'HistogramDisplayStyle','smooth','MarkerStyle','.','MarkerSize',50,'MarkerFilled','off','LegendVisible','on');


end