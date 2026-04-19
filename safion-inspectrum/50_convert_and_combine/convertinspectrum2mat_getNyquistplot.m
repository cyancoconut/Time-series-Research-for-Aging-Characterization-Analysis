%% Convert inspectrum to mat
file = 'C:\Users\sce\Desktop\EIS_Messungen\Safion_EIS_25C_ARN26650m1B_1_n_JMH_221114_00001.irf';

S = readstruct(file,'FileType','xml');
measurement_time = strcat(S.measurementResults.impedanceSpectrum.dateAttribute, " ", string(S.measurementResults.impedanceSpectrum.timeAttribute));
measurement_time = datetime(measurement_time,'InputFormat','yyyy/MM/dd HH:mm:ss','Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin');
voltage = S.measurementResults.batteryVoltage.voltageAttribute;
temperature = S.measurementResults.temperatureSensor.temperatureAttribute;

measurement = struct2table(S.measurementResults.impedanceSpectrum.spectrumData);
EIS_Frequency = measurement.freqAttribute;
EIS_Z_abs = abs(measurement.zReAttribute + 1i*measurement.zImAttribute);
EIS_Z_phase = angle(measurement.zReAttribute + 1i*measurement.zImAttribute);
EIS_Z = measurement.zReAttribute + 1i*measurement.zImAttribute;

measurement_time = (milliseconds(1:length(EIS_Frequency))/1000).'+measurement_time;

Current = ones(length(EIS_Frequency),1)*S.measurementResults.impedanceSpectrum.currentAverageAttribute;
Voltage = ones(length(EIS_Frequency),1)*voltage;
Temperature = ones(length(EIS_Frequency),1)*temperature;
AH_throughput = zeros(length(EIS_Frequency),1);
EIS_measurement_id = ones(length(EIS_Frequency),1);

TT_eis = [timetable(datetime(measurement_time,'Format','yyyy-MM-dd HH:mm:ss.SSSSSS','TimeZone','Europe/Berlin'),Current,Voltage,Temperature,EIS_Frequency,EIS_Z_abs,EIS_Z_phase,AH_throughput,EIS_measurement_id,'VariableNames',{'Current','Voltage','Temperature','EIS_Frequency','EIS_Z_abs','EIS_Z_phase','AH_throughput','EIS_measurement_id'})];

save(strcat(file(1:end-4),'.mat'),'TT_eis','-v7.3');
%% Nyquist plot

EIS_f = TT_eis.EIS_Frequency;
EIS_re = TT_eis.EIS_Z_abs.* cos(TT_eis.EIS_Z_phase);
EIS_imag = TT_eis.EIS_Z_abs .* sin(TT_eis.EIS_Z_phase);
% extract the data needed and filter for imag < 0
% EIS_f = EIS_f(EIS_imag < 0);
% EIS_re_1 = EIS_re(EIS_imag < 0);
% EIS_imag_1 = EIS_imag(EIS_imag < 0);

figure(1)
plot(EIS_re,EIS_imag,'-o','linewidth',1.5 );
xlabel('Real Impedance');ylabel('Imaginary Impedance');title('Nyquist Plot');
set(gca, 'YDir','reverse');
grid on
dcm.Enable = 'on';
dcm.DisplayStyle = 'window';