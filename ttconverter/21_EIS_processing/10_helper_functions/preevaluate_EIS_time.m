function [] = preevaluate_EIS_time(varargin)

close all

cell_name = "LiFun_575166-01";
path_to_data = "./../../../data/";
save_plots = 1;


figure_width_cm = 38;
figure_height_cm = 21;
FontSizePlots = 10;
FontNamePlots = "arial";

nan_threshold_s = 125;


%% config arguments
for i = 1:2:length(varargin)
    eval(strcat(varargin{i},"=""",varargin{i+1},""";"))
end


title_name = replace(cell_name,"_"," ");
% add helper functions to path
addpath("./10_helper_functions");

% create all necessary folders normally they should already be there
mkdir(strcat(path_to_data,"timeseries/"));
mkdir(strcat(path_to_data,"figures/"));
mkdir(strcat(path_to_data,"eis_data/"));

% load the EIS timeseries 
try
    load(strcat(path_to_data,'timeseries\',cell_name,'.mat'),'TT');
    TT.EIS_Frequency(1); % test if data is accessible
catch
    warning(strcat("EIS related timeseries file could not be load: ", strcat(path_to_data,'timeseries\',cell_name,'.mat') ));
    return;
end

try
    TT = removevars(TT,{'Prozedur'});
end
try
    TT = removevars(TT,{'Zustand'});
end


fig = figure();
set(fig,'Units','centimeters')
set(fig,'Position',[1,1,figure_width_cm,figure_height_cm]);
set(gca,'FontSize',FontSizePlots)
set(gca,'FontName',FontNamePlots)

indexNaN = find(seconds(diff(TT.Time)) > nan_threshold_s);

TT{indexNaN,:} = NaN;

TT(TT.Time < datetime("1980-01-01 00:00:00",'TimeZone','Europe/Berlin'),:) = [];

s = stackedplot(TT);

s.DisplayLabels = {{'Current in A','Ah Counter in Ah','Voltage in V','Temperature in $^\circ$C','Duration in days', 'Ah throughput in Ah','Capacity in Ah','Cur. of cap.-test in A'}};

ax = findobj(s.NodeChildren, 'Type','Axes');
set([ax(1).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')
set([ax(2).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')
set([ax(3).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')
% ax(3).YScale = 'log';
set([ax(4).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')
set([ax(5).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')
set([ax(6).YLabel],'Rotation',90,'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom','FontSize',FontSizePlots,'fontname',FontNamePlots,'interpreter','latex')

% sgtitle(convertStringsToChars(strcat('\textbf{',title_name,'}')),'Interpreter','latex','FontSize',FontSizePlots,'fontname',FontNamePlots);

xlabel('Date of Measurement')

for i = 1:6
    ax(i).XGrid = 'on';
    ax(i).YGrid = 'on';
    set(ax(i).XLabel,'Interpreter','latex');
    set(ax(i).YLabel,'Interpreter','latex');
    set(ax(i),'FontSize',FontSizePlots)
    set(ax(i),'FontName',FontNamePlots)
    ax(i).TickLabelInterpreter = 'latex';
    if save_plots
        if exist('exportgraphics')
            set(ax(i), 'color', 'none');
        end
    end
end



grid on


if save_plots
    
    file_name = strcat(replace(cell_name," ","_"),...
    "_eis_pre-evaluation_time");
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

end