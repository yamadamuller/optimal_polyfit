clc; clear all;

rawdata = readmatrix('./data/ICE.csv');
rawdata = rawdata(1:end,3:end);
rawdata = squeeze(mean(rawdata));
idx_mag = 1:2:length(rawdata);
mag = rawdata(idx_mag);
idx_phase = 2:2:length(rawdata);
phase = rawdata(idx_phase);
freq_response = mag.*exp(1j*phase);
freqs = logspace(log10(40), 6, 201);
ts = 0;
data = idfrd(squeeze(freq_response), squeeze(freqs), ts);
raw_fit = tfest(data,2,2); %3 poles, 3 zeros TF

bodemag(data, raw_fit);
xlim([40, 1e6])
legend('Measured','Estimated')
