function generate_simulated_sdr_dataset
clear; clc; close all;

%% ============================================================
% SIMULATED SDR DATASET FOR RF SIGNAL CLASSIFICATION
% WiFi / Bluetooth / Coexistence / Noise / Unknown
%
% This script simulates USRP-like I/Q captures and converts them
% into spectrogram images compatible with the trained CNN model.
%% ============================================================

projectDir = pwd;

baseDir = fullfile(projectDir, "data_simulated_sdr_v1");
specRoot = fullfile(baseDir, "spectrograms");
rawRoot  = fullfile(baseDir, "raw_iq_samples");

classes = ["Bluetooth", ...
           "Noise", ...
           "Unknown", ...
           "WiFi", ...
           "WiFi_Bluetooth_Overlap", ...
           "WiFi_Bluetooth_Separated"];

%% Dataset size
numExamplesPerClass = 600;      % 600 x 6 = 3600 spectrograms
saveRawIQ = true;               % true: save some I/Q examples
numRawPerClass = 25;            % save only 25 raw I/Q captures per class

%% Virtual SDR parameters
centerFrequency = 2.437e9;      % 2.437 GHz, WiFi channel 6 region
Fs = 40e6;                      % same sample rate used in training
Ntarget = 81920;                % 2.048 ms at 40 MS/s
imageSize = [224 224];

resetDataset = true;
rng(407);

%% Clean and create folders
if resetDataset && exist(baseDir, "dir")
    fprintf("Removing previous simulated SDR dataset:\n%s\n", baseDir);
    rmdir(baseDir, "s");
end

for c = classes
    mkdir(fullfile(specRoot, c));
    if saveRawIQ
        mkdir(fullfile(rawRoot, c));
    end
end

%% Metadata initialization
totalExamples = numExamplesPerClass * numel(classes);

metaFile = strings(totalExamples,1);
metaRawFile = strings(totalExamples,1);
metaClass = strings(totalExamples,1);
metaScenario = strings(totalExamples,1);
metaCenterFreqHz = nan(totalExamples,1);
metaSampleRateHz = nan(totalExamples,1);
metaGainDb = nan(totalExamples,1);
metaSNRdB = nan(totalExamples,1);
metaFreqOffsetMHz = nan(totalExamples,1);
metaClipRatio = nan(totalExamples,1);

row = 0;

fprintf("\nGenerating simulated SDR dataset...\n");
fprintf("Output folder: %s\n", baseDir);
fprintf("Examples per class: %d\n", numExamplesPerClass);
fprintf("Total spectrograms: %d\n\n", totalExamples);

%% ============================================================
% MAIN LOOP
%% ============================================================

for n = 1:numExamplesPerClass

    %% 1. Bluetooth
    label = "Bluetooth";
    snrDb = randomSNR();
    btSig = generateBluetoothSignal(Fs, Ntarget);
    btOffset = (-17 + 34*rand)*1e6;
    txSig = frequencyShift(btSig, Fs, btOffset);

    [rxWave, rxInfo] = simulateSDRReceiver(txSig, Fs, snrDb);
    saveExample(rxWave, rxInfo, label, "Bluetooth SDR-like capture", ...
        n, btOffset, snrDb);

    %% 2. Noise
    label = "Noise";
    snrDb = NaN;
    txSig = generateReceiverNoiseOnly(Fs, Ntarget);

    [rxWave, rxInfo] = simulateSDRNoiseCapture(txSig, Fs);
    saveExample(rxWave, rxInfo, label, "Receiver noise floor", ...
        n, NaN, snrDb);

    %% 3. Unknown
    label = "Unknown";
    snrDb = randomSNR();
    unkSig = generateUnknownSignal(Fs, Ntarget);
    unkOffset = (-18 + 36*rand)*1e6;
    txSig = frequencyShift(unkSig, Fs, unkOffset);

    [rxWave, rxInfo] = simulateSDRReceiver(txSig, Fs, snrDb);
    saveExample(rxWave, rxInfo, label, "Unknown SDR-like RF signal", ...
        n, unkOffset, snrDb);

    %% 4. WiFi
    label = "WiFi";
    snrDb = randomSNR();
    wifiSig = generateWiFiSignal(Fs, Ntarget);

    % Small random center mismatch inside the observed band
    wifiOffset = (-3 + 6*rand)*1e6;
    txSig = frequencyShift(wifiSig, Fs, wifiOffset);

    [rxWave, rxInfo] = simulateSDRReceiver(txSig, Fs, snrDb);
    saveExample(rxWave, rxInfo, label, "WiFi SDR-like capture", ...
        n, wifiOffset, snrDb);

    %% 5. WiFi + Bluetooth Overlap
    label = "WiFi_Bluetooth_Overlap";
    snrDb = randomSNR();

    wifiSig = generateWiFiSignal(Fs, Ntarget);
    btSig = generateBluetoothSignal(Fs, Ntarget);

    btOffset = (-8 + 16*rand)*1e6;
    btSig = frequencyShift(btSig, Fs, btOffset);

    btPower = randomRelativePower();
    txSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);

    [rxWave, rxInfo] = simulateSDRReceiver(txSig, Fs, snrDb);
    saveExample(rxWave, rxInfo, label, "WiFi and Bluetooth overlapping", ...
        n, btOffset, snrDb);

    %% 6. WiFi + Bluetooth Separated
    label = "WiFi_Bluetooth_Separated";
    snrDb = randomSNR();

    wifiSig = generateWiFiSignal(Fs, Ntarget);
    btSig = generateBluetoothSignal(Fs, Ntarget);

    if rand < 0.5
        btOffset = (-19 + 5*rand)*1e6;    % -19 to -14 MHz
    else
        btOffset = (14 + 5*rand)*1e6;     % +14 to +19 MHz
    end

    btSig = frequencyShift(btSig, Fs, btOffset);

    btPower = randomRelativePower();
    txSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);

    [rxWave, rxInfo] = simulateSDRReceiver(txSig, Fs, snrDb);
    saveExample(rxWave, rxInfo, label, "WiFi and Bluetooth separated", ...
        n, btOffset, snrDb);

    if mod(n,50) == 0 || n == 1
        fprintf("Progress: %d / %d per class\n", n, numExamplesPerClass);
    end
end

%% Save metadata
metaFile = metaFile(1:row);
metaRawFile = metaRawFile(1:row);
metaClass = metaClass(1:row);
metaScenario = metaScenario(1:row);
metaCenterFreqHz = metaCenterFreqHz(1:row);
metaSampleRateHz = metaSampleRateHz(1:row);
metaGainDb = metaGainDb(1:row);
metaSNRdB = metaSNRdB(1:row);
metaFreqOffsetMHz = metaFreqOffsetMHz(1:row);
metaClipRatio = metaClipRatio(1:row);

metadata = table(metaFile(:), metaRawFile(:), metaClass(:), metaScenario(:), ...
    metaCenterFreqHz(:), metaSampleRateHz(:), metaGainDb(:), metaSNRdB(:), ...
    metaFreqOffsetMHz(:), metaClipRatio(:));

metadata.Properties.VariableNames = { ...
    'SpectrogramFile', ...
    'RawIQFile', ...
    'Class', ...
    'Scenario', ...
    'CenterFrequency_Hz', ...
    'SampleRate_Hz', ...
    'ReceiverGain_dB', ...
    'SNR_dB', ...
    'FrequencyOffset_MHz', ...
    'ADC_ClipRatio'};

metadataPath = fullfile(baseDir, "metadata_simulated_sdr_v1.csv");
writetable(metadata, metadataPath);

fprintf("\nSimulated SDR dataset generated successfully.\n");
fprintf("Metadata saved to:\n%s\n", metadataPath);

%% Verify dataset
imds = imageDatastore(specRoot, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Spectrogram count per class:");
disp(countEachLabel(imds));

figure;
idx = randperm(numel(imds.Files), min(30,numel(imds.Files)));
montage(imds.Files(idx));
title("Simulated SDR spectrogram samples");

%% ============================================================
% NESTED SAVE FUNCTION
%% ============================================================

function saveExample(rxWave, rxInfo, label, scenario, n, freqOffsetHz, snrDb)

    % Access variables from parent workspace
    specFolder = fullfile(specRoot, label);
    rawFolder = fullfile(rawRoot, label);

    img = iqToSpectrogramImage(rxWave, Fs, imageSize);

    specFile = fullfile(specFolder, sprintf("%s_sdrsim_%05d.png", label, n));
    imwrite(img, specFile);

    rawFile = "";

    if saveRawIQ && n <= numRawPerClass
        rawFile = fullfile(rawFolder, sprintf("%s_sdrsim_iq_%05d.mat", label, n));

        metadataRaw = struct();
        metadataRaw.label = label;
        metadataRaw.scenario = scenario;
        metadataRaw.centerFrequency = centerFrequency;
        metadataRaw.sampleRate = Fs;
        metadataRaw.Ntarget = Ntarget;
        metadataRaw.receiverGainDb = rxInfo.gainDb;
        metadataRaw.snrDb = snrDb;
        metadataRaw.frequencyOffsetHz = freqOffsetHz;
        metadataRaw.clipRatio = rxInfo.clipRatio;
        metadataRaw.dateTime = datetime("now");

        save(rawFile, "rxWave", "metadataRaw");
    end

    row = row + 1;

    metaFile(row) = string(specFile);
    metaRawFile(row) = string(rawFile);
    metaClass(row) = string(label);
    metaScenario(row) = string(scenario);
    metaCenterFreqHz(row) = centerFrequency;
    metaSampleRateHz(row) = Fs;
    metaGainDb(row) = rxInfo.gainDb;
    metaSNRdB(row) = snrDb;

    if isnan(freqOffsetHz)
        metaFreqOffsetMHz(row) = NaN;
    else
        metaFreqOffsetMHz(row) = freqOffsetHz/1e6;
    end

    metaClipRatio(row) = rxInfo.clipRatio;
end

%% ============================================================
% HELPER FUNCTIONS
%% ============================================================

function sig = generateWiFiSignal(FsTarget, Ntarget)
    cfgWiFi = wlanNonHTConfig;
    cfgWiFi.ChannelBandwidth = "CBW20";
    cfgWiFi.MCS = randi([0 7]);
    cfgWiFi.PSDULength = randi([200 2200]);

    bitsWiFi = randi([0 1], cfgWiFi.PSDULength*8, 1);
    sig = wlanWaveformGenerator(bitsWiFi, cfgWiFi);

    FsWiFi = wlanSampleRate(cfgWiFi);
    sig = resample(sig, FsTarget, FsWiFi);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);

    % SDR-like packet activity in part of the cases
    if rand < 0.35
        sig = applyBurstMask(sig);
    end
end

function sig = generateBluetoothSignal(FsTarget, Ntarget)
    sps = 8;
    FsBT = 1e6 * sps;

    msgLen = randi([250 2080]);
    messageBT = randi([0 1], msgLen, 1);

    sig = bleWaveformGenerator(messageBT, ...
        "Mode", "LE1M", ...
        "SamplesPerSymbol", sps);

    sig = resample(sig, FsTarget, FsBT);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);

    % Bluetooth often appears as short activity bursts
    if rand < 0.60
        sig = applyBurstMask(sig);
    end
end

function sig = generateUnknownSignal(Fs, Ntarget)
    signalType = randi([1 7]);

    switch signalType
        case 1
            % BPSK-like
            sps = randi([8 30]);
            numSymbols = ceil(Ntarget/sps) + 100;
            bits = randi([0 1], numSymbols, 1);
            symbols = 2*bits - 1;
            sig = complex(repelem(symbols, sps), 0);

        case 2
            % QPSK-like
            sps = randi([6 26]);
            numSymbols = ceil(Ntarget/sps) + 100;
            data = randi([0 3], numSymbols, 1);
            symbols = exp(1j*(pi/4 + data*pi/2));
            sig = repelem(symbols, sps);

        case 3
            % Chirp
            t = (0:Ntarget-1).' / Fs;
            f0 = (-7 + 14*rand)*1e6;
            f1 = (-7 + 14*rand)*1e6;
            k = (f1 - f0) / max(t);
            phase = 2*pi*(f0*t + 0.5*k*t.^2);
            sig = exp(1j*phase);

        case 4
            % Multitone
            t = (0:Ntarget-1).' / Fs;
            sig = zeros(Ntarget,1);
            numTones = randi([2 6]);
            for k = 1:numTones
                f = (-7 + 14*rand)*1e6;
                amp = 0.2 + 0.9*rand;
                sig = sig + amp*exp(1j*2*pi*f*t);
            end

        case 5
            % FSK-like
            sps = randi([6 22]);
            numSymbols = ceil(Ntarget/sps) + 100;
            bits = randi([0 1], numSymbols, 1);
            freqDev = (0.2 + 1.6*rand)*1e6;

            freqSeq = (2*bits - 1)*freqDev;
            freqSamples = repelem(freqSeq, sps);
            freqSamples = repeatOrTrim(freqSamples, Ntarget);

            phase = cumsum(2*pi*freqSamples/Fs);
            sig = exp(1j*phase);

        case 6
            % Non-WiFi OFDM-like signal
            nfft = 256;
            cpLen = 32;
            numSym = ceil(Ntarget/(nfft+cpLen)) + 10;
            activeBins = 30 + randi([0 110]);

            sig = [];

            for m = 1:numSym
                X = zeros(nfft,1);
                idxStart = floor(nfft/2) - floor(activeBins/2) + 1;
                idxEnd = idxStart + activeBins - 1;

                qpskData = randi([0 3], activeBins, 1);
                symbols = exp(1j*(pi/4 + qpskData*pi/2));

                X(idxStart:idxEnd) = symbols;
                x = ifft(ifftshift(X));
                xcp = [x(end-cpLen+1:end); x];

                sig = [sig; xcp]; %#ok<AGROW>
            end

        case 7
            % Pulsed burst signal
            sig = zeros(Ntarget,1);
            numBursts = randi([2 8]);

            for b = 1:numBursts
                burstLen = randi([round(0.02*Ntarget) round(0.12*Ntarget)]);
                startIdx = randi([1 Ntarget-burstLen+1]);

                sps = randi([6 20]);
                numSymbols = ceil(burstLen/sps) + 10;
                data = randi([0 3], numSymbols, 1);
                symbols = exp(1j*(pi/4 + data*pi/2));
                burst = repelem(symbols, sps);
                burst = repeatOrTrim(burst, burstLen);

                sig(startIdx:startIdx+burstLen-1) = ...
                    sig(startIdx:startIdx+burstLen-1) + burst;
            end
    end

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateReceiverNoiseOnly(Fs, Ntarget)
    sig = complex(randn(Ntarget,1), randn(Ntarget,1));

    % Colored receiver noise
    if rand < 0.80
        filtLen = randi([3 25]);
        h = rand(filtLen,1);
        h = h / sum(h);
        sig = filter(h, 1, sig);
    end

    % Weak spur
    if rand < 0.35
        t = (0:Ntarget-1).' / Fs;
        fspur = (-18 + 36*rand)*1e6;
        sig = sig + 0.03*exp(1j*2*pi*fspur*t);
    end

    sig = normalizeSignal(sig);
end

function [rx, info] = simulateSDRReceiver(tx, Fs, snrDb)
    tx = normalizeSignal(tx);

    % Random phase
    rx = tx * exp(1j*2*pi*rand);

    % Fine oscillator offset
    fineOffset = (-1.2 + 2.4*rand)*1e6;
    rx = frequencyShift(rx, Fs, fineOffset);

    % Multipath
    if rand < 0.75
        nTaps = randi([2 12]);
        decay = exp(-(0:nTaps-1).'/3);
        h = (randn(nTaps,1) + 1j*randn(nTaps,1)).*decay;
        h = h / norm(h);
        rx = filter(h, 1, rx);
    end

    % IQ imbalance
    if rand < 0.85
        rx = applyIQImbalance(rx);
    end

    % DC offset / LO leakage
    if rand < 0.70
        dc = (0.005 + 0.06*rand)*exp(1j*2*pi*rand);
        rx = rx + dc;
    end

    % Add receiver noise
    rx = normalizeSignal(rx);
    rx = awgn(rx, snrDb, "measured");

    % Receiver gain and ADC quantization
    gainDb = -8 + 16*rand;
    [rx, clipRatio] = simulateADC(rx, gainDb);

    info.gainDb = gainDb;
    info.clipRatio = clipRatio;
end

function [rx, info] = simulateSDRNoiseCapture(noiseIn, Fs)
    rx = normalizeSignal(noiseIn);

    % Add DC/LO leakage
    if rand < 0.75
        rx = rx + (0.005 + 0.07*rand)*exp(1j*2*pi*rand);
    end

    % Add low-level center spur
    if rand < 0.45
        t = (0:length(rx)-1).' / Fs;
        rx = rx + 0.02*exp(1j*2*pi*(0.05e6*randn)*t);
    end

    gainDb = -10 + 14*rand;
    [rx, clipRatio] = simulateADC(rx, gainDb);

    info.gainDb = gainDb;
    info.clipRatio = clipRatio;
end

function [y, clipRatio] = simulateADC(x, gainDb)
    x = normalizeSignal(x);

    gainLinear = 10^(gainDb/20);
    x = x * gainLinear;

    I = real(x);
    Q = imag(x);

    clipMask = abs(I) > 1 | abs(Q) > 1;
    clipRatio = mean(clipMask);

    I = max(min(I, 1), -1);
    Q = max(min(Q, 1), -1);

    % Quantize as signed int16, then dequantize
    Iq = int16(round(I * 32767));
    Qq = int16(round(Q * 32767));

    Irec = double(Iq) / 32768;
    Qrec = double(Qq) / 32768;

    y = single(complex(Irec, Qrec));
    y = y(:);
end

function y = applyIQImbalance(x)
    ampImbDb = -4.5 + 9*rand;
    amp = 10^(ampImbDb/20);

    phaseImb = (-5.5 + 11*rand) * pi/180;

    I = real(x);
    Q = imag(x);

    I2 = amp * I;
    Q2 = (1/amp) * Q;

    y = I2 + 1j*(Q2*cos(phaseImb) + I2*sin(phaseImb));
end

function y = applyBurstMask(x)
    N = length(x);
    mask = zeros(N,1);

    numBursts = randi([1 5]);

    for b = 1:numBursts
        burstLen = randi([round(0.08*N) round(0.45*N)]);
        startIdx = randi([1 N-burstLen+1]);

        fade = hann(min(128, burstLen));
        burstMask = ones(burstLen,1);

        L = length(fade);
        halfL = floor(L/2);

        if burstLen > L
            burstMask(1:halfL) = fade(1:halfL);
            burstMask(end-halfL+1:end) = fade(end-halfL+1:end);
        end

        mask(startIdx:startIdx+burstLen-1) = ...
            max(mask(startIdx:startIdx+burstLen-1), burstMask);
    end

    y = x(:).*mask;
    y = normalizeSignal(y);
end

function y = frequencyShift(x, Fs, freqOffset)
    t = (0:length(x)-1).' / Fs;
    y = x(:).*exp(1j*2*pi*freqOffset*t);
end

function snrDb = randomSNR()
    snrList = [-6 -4 -2 0 2 4 6 8 10 12 15 18 20 25 30];
    snrDb = snrList(randi(numel(snrList)));
end

function p = randomRelativePower()
    powerList = [0.08 0.12 0.15 0.20 0.30 0.40 0.55 0.70 0.90 1.10 1.30 1.50];
    p = powerList(randi(numel(powerList)));
end

function img = iqToSpectrogramImage(sig, Fs, imageSize)
    sig = sig(:);
    sig = normalizeSignal(sig);

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

function y = normalizeSignal(x)
    y = x ./ max(abs(x) + eps);
end

end