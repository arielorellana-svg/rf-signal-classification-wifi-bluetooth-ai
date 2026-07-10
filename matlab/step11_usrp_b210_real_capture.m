clear; clc; close all;

%% ============================================================
% USRP B210 real RF capture dataset
% Passive reception only
%
% Captures raw I/Q and spectrogram images compatible with the CNN model.
%% ============================================================

%% Select capture label manually
label = "WiFi";
% Use one of:
% "Bluetooth"
% "Noise"
% "Unknown"
% "WiFi"
% "WiFi_Bluetooth_Overlap"
% "WiFi_Bluetooth_Separated"

scenario = "Real WiFi capture using USRP B210";
% Example scenarios:
% "Receiver noise floor"
% "Real WiFi capture using USRP B210"
% "Real Bluetooth activity near receiver"
% "Real WiFi and Bluetooth coexistence"

%% Project folders
projectDir = pwd;

baseDir = fullfile(projectDir, "data_sdr");
rawRoot = fullfile(baseDir, "raw_iq", label);
specRoot = fullfile(baseDir, "spectrograms", label);
metadataPath = fullfile(baseDir, "metadata_real_sdr_b210.csv");

if ~exist(rawRoot, "dir")
    mkdir(rawRoot);
end

if ~exist(specRoot, "dir")
    mkdir(specRoot);
end

metadataPath = fullfile(baseDir, "metadata_real_sdr_b210.csv");

%% USRP parameters
platform = "B210";
serialNum = "";                    % Leave empty for auto-detect

centerFrequency = 2.437e9;         % 2.437 GHz, WiFi channel 6 region
masterClockRate = 40e6;
decimationFactor = 1;
Fs = masterClockRate / decimationFactor;

gain = 35;                         % Start with 30-40 dB
Ntarget = 81920;                   % 2.048 ms at 40 MS/s
imageSize = [224 224];

numCaptures = 600;                 % captures for this label
saveRawIQ = true;                  % save raw I/Q .mat files
pauseBetweenCaptures = 0.02;

%% Auto-detect B210 serial
radios = findsdru;
disp(radios);

if strlength(serialNum) == 0
    idx = find(strcmp({radios.Platform}, "B210") & strcmp({radios.Status}, "Success"), 1);
    if isempty(idx)
        error("No B210 with Status=Success was detected.");
    end
    serialNum = string(radios(idx).SerialNum);
end

fprintf("Using B210 serial: %s\n", serialNum);

%% Create receiver
rx = comm.SDRuReceiver( ...
    Platform=platform, ...
    SerialNum=serialNum, ...
    CenterFrequency=centerFrequency, ...
    MasterClockRate=masterClockRate, ...
    DecimationFactor=decimationFactor, ...
    SamplesPerFrame=Ntarget, ...
    Gain=gain, ...
    ChannelMapping=1, ...
    OutputDataType="double");

%% Warm-up frames
disp("Warming up receiver...");
for k = 1:10
    rx();
end

%% Metadata table initialization
metaSpectrogramFile = strings(numCaptures,1);
metaRawIQFile = strings(numCaptures,1);
metaClass = strings(numCaptures,1);
metaScenario = strings(numCaptures,1);
metaCenterFrequencyHz = nan(numCaptures,1);
metaSampleRateHz = nan(numCaptures,1);
metaGainDb = nan(numCaptures,1);
metaCaptureIndex = nan(numCaptures,1);
metaDataLength = nan(numCaptures,1);
metaOverrun = nan(numCaptures,1);
metaPowerDb = nan(numCaptures,1);
metaPeakDb = nan(numCaptures,1);
metaDateTime = strings(numCaptures,1);

validCount = 0;
overrunCount = 0;

fprintf("\nStarting USRP B210 captures for label: %s\n", label);
fprintf("Center frequency: %.3f GHz\n", centerFrequency/1e9);
fprintf("Sample rate: %.2f MS/s\n", Fs/1e6);
fprintf("Gain: %.1f dB\n\n", gain);

for k = 1:numCaptures

    [rxWave, len, overrun] = rx();

    if overrun ~= 0
        overrunCount = overrunCount + 1;
    end

    if len <= 0
        fprintf("Capture %d skipped: no valid data.\n", k);
        continue;
    end

    rxWave = rxWave(:);
    rxWave = rxWave(1:min(length(rxWave), Ntarget));
    rxWave = repeatOrTrim(rxWave, Ntarget);

    validCount = validCount + 1;

    %% Save spectrogram
    img = iqToSpectrogramImage(rxWave, Fs, imageSize);

    specFile = fullfile(specRoot, sprintf("%s_usrp_%05d.png", label, validCount));
    imwrite(img, specFile);

    %% Save raw I/Q
    rawFile = "";

    if saveRawIQ
        rawFile = fullfile(rawRoot, sprintf("%s_usrp_iq_%05d.mat", label, validCount));

        metadataRaw = struct();
        metadataRaw.label = label;
        metadataRaw.scenario = scenario;
        metadataRaw.centerFrequency = centerFrequency;
        metadataRaw.sampleRate = Fs;
        metadataRaw.masterClockRate = masterClockRate;
        metadataRaw.decimationFactor = decimationFactor;
        metadataRaw.gain = gain;
        metadataRaw.samplesPerFrame = Ntarget;
        metadataRaw.dataLength = len;
        metadataRaw.overrun = overrun;
        metadataRaw.dateTime = datetime("now");

        save(rawFile, "rxWave", "metadataRaw");
    end

    %% Compute simple metadata
    pwrDb = 10*log10(mean(abs(rxWave).^2) + eps);
    peakDb = 20*log10(max(abs(rxWave)) + eps);

    metaSpectrogramFile(validCount) = string(specFile);
    metaRawIQFile(validCount) = string(rawFile);
    metaClass(validCount) = label;
    metaScenario(validCount) = scenario;
    metaCenterFrequencyHz(validCount) = centerFrequency;
    metaSampleRateHz(validCount) = Fs;
    metaGainDb(validCount) = gain;
    metaCaptureIndex(validCount) = validCount;
    metaDataLength(validCount) = len;
    metaOverrun(validCount) = overrun;
    metaPowerDb(validCount) = pwrDb;
    metaPeakDb(validCount) = peakDb;
    metaDateTime(validCount) = string(datetime("now"));

    if mod(k,25) == 0 || k == 1
        fprintf("Capture %d/%d | valid=%d | overrun=%d | Pwr=%.2f dB\n", ...
            k, numCaptures, validCount, overrun, pwrDb);
    end

    pause(pauseBetweenCaptures);
end

release(rx);

%% Trim metadata
metaSpectrogramFile = metaSpectrogramFile(1:validCount);
metaRawIQFile = metaRawIQFile(1:validCount);
metaClass = metaClass(1:validCount);
metaScenario = metaScenario(1:validCount);
metaCenterFrequencyHz = metaCenterFrequencyHz(1:validCount);
metaSampleRateHz = metaSampleRateHz(1:validCount);
metaGainDb = metaGainDb(1:validCount);
metaCaptureIndex = metaCaptureIndex(1:validCount);
metaDataLength = metaDataLength(1:validCount);
metaOverrun = metaOverrun(1:validCount);
metaPowerDb = metaPowerDb(1:validCount);
metaPeakDb = metaPeakDb(1:validCount);
metaDateTime = metaDateTime(1:validCount);

metadata = table( ...
    metaSpectrogramFile(:), ...
    metaRawIQFile(:), ...
    metaClass(:), ...
    metaScenario(:), ...
    metaCenterFrequencyHz(:), ...
    metaSampleRateHz(:), ...
    metaGainDb(:), ...
    metaCaptureIndex(:), ...
    metaDataLength(:), ...
    metaOverrun(:), ...
    metaPowerDb(:), ...
    metaPeakDb(:), ...
    metaDateTime(:));

metadata.Properties.VariableNames = { ...
    'SpectrogramFile', ...
    'RawIQFile', ...
    'Class', ...
    'Scenario', ...
    'CenterFrequency_Hz', ...
    'SampleRate_Hz', ...
    'Gain_dB', ...
    'CaptureIndex', ...
    'DataLength', ...
    'Overrun', ...
    'Power_dB', ...
    'Peak_dB', ...
    'DateTime'};

%% Append or create metadata CSV
if exist(metadataPath, "file")
    oldMetadata = readtable(metadataPath);
    metadata = [oldMetadata; metadata];
end

writetable(metadata, metadataPath);

fprintf("\nUSRP capture completed.\n");
fprintf("Valid captures saved: %d\n", validCount);
fprintf("Overruns detected: %d\n", overrunCount);
fprintf("Spectrogram folder:\n%s\n", specRoot);
fprintf("Metadata file:\n%s\n", metadataPath);

%% Verify current dataset
imds = imageDatastore(fullfile(baseDir, "spectrograms"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Current real USRP dataset count:");
disp(countEachLabel(imds));

%% Helper functions
function img = iqToSpectrogramImage(sig, Fs, imageSize)
    sig = sig(:);
    sig = sig ./ max(abs(sig) + eps);

    win = hamming(1024);
    noverlap = 768;
    nfft = 2048;

    [S,~,~] = spectrogram(sig, win, noverlap, nfft, Fs, "centered");

    img = 20*log10(abs(S) + eps);
    img = img - max(img(:));
    img = max(img, -90);
    img = (img + 90) / 90;
    img = imresize(img, imageSize);
end

function y = repeatOrTrim(x, N)
    x = x(:);
    if length(x) < N
        reps = ceil(N/length(x));
        x = repmat(x, reps, 1);
    end
    y = x(1:N);
end
