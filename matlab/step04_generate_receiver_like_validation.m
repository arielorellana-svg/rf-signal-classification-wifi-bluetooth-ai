clearvars; clc; close all;

%% ============================================================
% STEP 04 - GENERATE INDEPENDENT RECEIVER-LIKE VALIDATION DATASET
% WiFi / Bluetooth / coexistence / noise / unknown RF activity
%
% The dataset is generated entirely from synthetic complex I/Q waveforms.
% A second impairment chain introduces acquisition-inspired effects such as
% oscillator mismatch, multipath, IQ imbalance, DC leakage, receiver gain,
% additive noise, clipping, and ADC quantization.
%% ============================================================

%% Project paths
scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

baseDir = fullfile(projectDir, "data", "receiver_like_validation");
specRoot = fullfile(baseDir, "spectrograms");
rawRoot = fullfile(baseDir, "raw_iq_samples");
resultsDir = fullfile(projectDir, "results");

classes = ["Bluetooth", ...
           "Noise", ...
           "Unknown", ...
           "WiFi", ...
           "WiFi_Bluetooth_Overlap", ...
           "WiFi_Bluetooth_Separated"];

%% Reproducible dataset configuration
datasetVersion = "receiver_like_v1";
randomSeed = 407;
numExamplesPerClass = 600;
saveRawIQ = false;
numRawPerClass = 25;
resetDataset = true;
createPreview = true;

%% Nominal observation parameters
nominalCenterFrequency = 2.437e9;
Fs = 40e6;
Ntarget = 81920;
imageSize = [224 224];

rng(randomSeed, "twister");

%% Prepare output folders
if resetDataset && exist(baseDir, "dir")
    fprintf("Removing previous receiver-like dataset:\n%s\n", baseDir);
    rmdir(baseDir, "s");
end

for c = classes
    specFolder = fullfile(specRoot, c);
    if ~exist(specFolder, "dir")
        mkdir(specFolder);
    end

    if saveRawIQ
        rawFolder = fullfile(rawRoot, c);
        if ~exist(rawFolder, "dir")
            mkdir(rawFolder);
        end
    end
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

%% Metadata initialization
totalExamples = numExamplesPerClass * numel(classes);
emptyRecord = struct( ...
    "SpectrogramFile", "", ...
    "RawIQFile", "", ...
    "Class", "", ...
    "Scenario", "", ...
    "DatasetVersion", datasetVersion, ...
    "RandomSeed", randomSeed, ...
    "NominalCenterFrequency_Hz", NaN, ...
    "SampleRate_Hz", NaN, ...
    "ReceiverGain_dB", NaN, ...
    "SNR_dB", NaN, ...
    "ProgrammedFrequencyOffset_MHz", NaN, ...
    "FineOscillatorOffset_MHz", NaN, ...
    "ADC_ClipRatio", NaN);
records = repmat(emptyRecord, totalExamples, 1);
row = 0;

fprintf("\nGenerating independent receiver-like validation dataset...\n");
fprintf("Output folder: %s\n", baseDir);
fprintf("Examples per class: %d\n", numExamplesPerClass);
fprintf("Total spectrograms: %d\n", totalExamples);
fprintf("Random seed: %d\n\n", randomSeed);

%% Main generation loop
for n = 1:numExamplesPerClass

    %% 1. Bluetooth
    label = "Bluetooth";
    snrDb = randomSNR();
    btSig = generateBluetoothSignal(Fs, Ntarget);
    btOffset = (-17 + 34*rand)*1e6;
    txSig = frequencyShift(btSig, Fs, btOffset);

    [rxWave, rxInfo] = simulateReceiver(txSig, Fs, snrDb);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "Bluetooth receiver-like observation", n, btOffset, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    %% 2. Noise
    label = "Noise";
    snrDb = NaN;
    txSig = generateReceiverNoiseOnly(Fs, Ntarget);

    [rxWave, rxInfo] = simulateNoiseObservation(txSig, Fs);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "Receiver-like noise floor", n, NaN, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    %% 3. Unknown
    label = "Unknown";
    snrDb = randomSNR();
    unknownSig = generateUnknownSignal(Fs, Ntarget);
    unknownOffset = (-18 + 36*rand)*1e6;
    txSig = frequencyShift(unknownSig, Fs, unknownOffset);

    [rxWave, rxInfo] = simulateReceiver(txSig, Fs, snrDb);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "Unknown receiver-like RF observation", n, unknownOffset, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    %% 4. WiFi
    label = "WiFi";
    snrDb = randomSNR();
    wifiSig = generateWiFiSignal(Fs, Ntarget);
    wifiOffset = (-3 + 6*rand)*1e6;
    txSig = frequencyShift(wifiSig, Fs, wifiOffset);

    [rxWave, rxInfo] = simulateReceiver(txSig, Fs, snrDb);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "WiFi receiver-like observation", n, wifiOffset, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    %% 5. WiFi + Bluetooth overlap
    label = "WiFi_Bluetooth_Overlap";
    snrDb = randomSNR();
    wifiSig = generateWiFiSignal(Fs, Ntarget);
    btSig = generateBluetoothSignal(Fs, Ntarget);
    btOffset = (-8 + 16*rand)*1e6;
    btSig = frequencyShift(btSig, Fs, btOffset);
    btPower = randomRelativePower();
    txSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);

    [rxWave, rxInfo] = simulateReceiver(txSig, Fs, snrDb);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "WiFi and Bluetooth overlapping", n, btOffset, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    %% 6. WiFi + Bluetooth separated
    label = "WiFi_Bluetooth_Separated";
    snrDb = randomSNR();
    wifiSig = generateWiFiSignal(Fs, Ntarget);
    btSig = generateBluetoothSignal(Fs, Ntarget);

    if rand < 0.5
        btOffset = (-19 + 5*rand)*1e6;
    else
        btOffset = (14 + 5*rand)*1e6;
    end

    btSig = frequencyShift(btSig, Fs, btOffset);
    btPower = randomRelativePower();
    txSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);

    [rxWave, rxInfo] = simulateReceiver(txSig, Fs, snrDb);
    row = row + 1;
    records(row) = saveReceiverLikeExample(rxWave, rxInfo, label, ...
        "WiFi and Bluetooth separated", n, btOffset, snrDb, ...
        projectDir, specRoot, rawRoot, Fs, Ntarget, imageSize, ...
        nominalCenterFrequency, saveRawIQ, numRawPerClass, ...
        datasetVersion, randomSeed);

    if mod(n,50) == 0 || n == 1
        fprintf("Progress: %d / %d per class\n", n, numExamplesPerClass);
    end
end

%% Save metadata and generation configuration
records = records(1:row);
metadata = struct2table(records);
metadataPath = fullfile(baseDir, "metadata_receiver_like_validation.csv");
writetable(metadata, metadataPath);

configuration = table( ...
    datasetVersion, randomSeed, numExamplesPerClass, numel(classes), ...
    nominalCenterFrequency, Fs, Ntarget, imageSize(1), imageSize(2), ...
    saveRawIQ, numRawPerClass, ...
    'VariableNames', { ...
        'DatasetVersion', 'RandomSeed', 'ExamplesPerClass', 'NumClasses', ...
        'NominalCenterFrequency_Hz', 'SampleRate_Hz', 'SamplesPerObservation', ...
        'ImageHeight', 'ImageWidth', 'SaveRawIQ', 'RawExamplesPerClass'});
configurationPath = fullfile(baseDir, "generation_config_receiver_like.csv");
writetable(configuration, configurationPath);

fprintf("\nReceiver-like validation dataset generated successfully.\n");
fprintf("Metadata: %s\n", metadataPath);
fprintf("Configuration: %s\n", configurationPath);

%% Verify dataset structure
imds = imageDatastore(specRoot, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");
counts = countEachLabel(imds);
disp("Spectrogram count per class:");
disp(counts);

if height(counts) ~= numel(classes) || any(counts.Count ~= numExamplesPerClass)
    error("Generated dataset does not contain the expected number of images per class.");
end

if createPreview
    fig = figure;
    idx = randperm(numel(imds.Files), min(30,numel(imds.Files)));
    montage(imds.Files(idx));
    title("Receiver-Like Validation Spectrogram Samples");
    exportgraphics(fig, fullfile(resultsDir, ...
        "receiver_like_dataset_preview.png"), "Resolution", 200);
end

%% Local functions
function record = saveReceiverLikeExample(rxWave, rxInfo, label, scenario, ...
        n, programmedOffsetHz, snrDb, projectDir, specRoot, rawRoot, ...
        Fs, Ntarget, imageSize, nominalCenterFrequency, saveRawIQ, ...
        numRawPerClass, datasetVersion, randomSeed)

    specFolder = fullfile(specRoot, label);
    img = iqToSpectrogramImage(rxWave, Fs, imageSize);

    specFile = fullfile(specFolder, ...
        sprintf("%s_receiverlike_%05d.png", label, n));
    imwrite(img, specFile);

    rawFile = "";
    if saveRawIQ && n <= numRawPerClass
        rawFolder = fullfile(rawRoot, label);
        rawFile = fullfile(rawFolder, ...
            sprintf("%s_receiverlike_iq_%05d.mat", label, n));

        metadataRaw = struct();
        metadataRaw.datasetVersion = datasetVersion;
        metadataRaw.randomSeed = randomSeed;
        metadataRaw.label = label;
        metadataRaw.scenario = scenario;
        metadataRaw.nominalCenterFrequency = nominalCenterFrequency;
        metadataRaw.sampleRate = Fs;
        metadataRaw.samplesPerObservation = Ntarget;
        metadataRaw.receiverGainDb = rxInfo.gainDb;
        metadataRaw.snrDb = snrDb;
        metadataRaw.programmedFrequencyOffsetHz = programmedOffsetHz;
        metadataRaw.fineOscillatorOffsetHz = rxInfo.fineOffsetHz;
        metadataRaw.clipRatio = rxInfo.clipRatio;

        save(rawFile, "rxWave", "metadataRaw", "-v7.3");
    end

    record = struct( ...
        "SpectrogramFile", relativePath(projectDir, specFile), ...
        "RawIQFile", relativePath(projectDir, rawFile), ...
        "Class", string(label), ...
        "Scenario", string(scenario), ...
        "DatasetVersion", datasetVersion, ...
        "RandomSeed", randomSeed, ...
        "NominalCenterFrequency_Hz", nominalCenterFrequency, ...
        "SampleRate_Hz", Fs, ...
        "ReceiverGain_dB", rxInfo.gainDb, ...
        "SNR_dB", snrDb, ...
        "ProgrammedFrequencyOffset_MHz", toMHz(programmedOffsetHz), ...
        "FineOscillatorOffset_MHz", toMHz(rxInfo.fineOffsetHz), ...
        "ADC_ClipRatio", rxInfo.clipRatio);
end

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

    if rand < 0.35
        sig = applyBurstMask(sig);
    end
end

function sig = generateBluetoothSignal(FsTarget, Ntarget)
    samplesPerSymbol = 8;
    FsBluetooth = 1e6 * samplesPerSymbol;
    messageLength = randi([250 2080]);
    message = randi([0 1], messageLength, 1);

    sig = bleWaveformGenerator(message, ...
        "Mode", "LE1M", ...
        "SamplesPerSymbol", samplesPerSymbol);

    sig = resample(sig, FsTarget, FsBluetooth);
    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);

    if rand < 0.60
        sig = applyBurstMask(sig);
    end
end

function sig = generateUnknownSignal(Fs, Ntarget)
    signalType = randi([1 7]);

    switch signalType
        case 1
            samplesPerSymbol = randi([8 30]);
            numSymbols = ceil(Ntarget/samplesPerSymbol) + 100;
            bits = randi([0 1], numSymbols, 1);
            symbols = 2*bits - 1;
            repeatedSymbols = repelem(symbols, samplesPerSymbol);
            sig = complex(repeatedSymbols, zeros(size(repeatedSymbols)));

        case 2
            samplesPerSymbol = randi([6 26]);
            numSymbols = ceil(Ntarget/samplesPerSymbol) + 100;
            data = randi([0 3], numSymbols, 1);
            symbols = exp(1j*(pi/4 + data*pi/2));
            sig = repelem(symbols, samplesPerSymbol);

        case 3
            t = (0:Ntarget-1).' / Fs;
            f0 = (-7 + 14*rand)*1e6;
            f1 = (-7 + 14*rand)*1e6;
            chirpRate = (f1 - f0) / max(t);
            phase = 2*pi*(f0*t + 0.5*chirpRate*t.^2);
            sig = exp(1j*phase);

        case 4
            t = (0:Ntarget-1).' / Fs;
            sig = zeros(Ntarget,1);
            numTones = randi([2 6]);
            for k = 1:numTones
                frequency = (-7 + 14*rand)*1e6;
                amplitude = 0.2 + 0.9*rand;
                sig = sig + amplitude*exp(1j*2*pi*frequency*t);
            end

        case 5
            samplesPerSymbol = randi([6 22]);
            numSymbols = ceil(Ntarget/samplesPerSymbol) + 100;
            bits = randi([0 1], numSymbols, 1);
            frequencyDeviation = (0.2 + 1.6*rand)*1e6;
            frequencySequence = (2*bits - 1)*frequencyDeviation;
            frequencySamples = repelem(frequencySequence, samplesPerSymbol);
            frequencySamples = repeatOrTrim(frequencySamples, Ntarget);
            phase = cumsum(2*pi*frequencySamples/Fs);
            sig = exp(1j*phase);

        case 6
            nfft = 256;
            cyclicPrefixLength = 32;
            numSymbols = ceil(Ntarget/(nfft+cyclicPrefixLength)) + 10;
            activeBins = 30 + randi([0 110]);
            sig = complex(zeros(0,1));

            for m = 1:numSymbols
                X = zeros(nfft,1);
                idxStart = floor(nfft/2) - floor(activeBins/2) + 1;
                idxEnd = idxStart + activeBins - 1;
                qpskData = randi([0 3], activeBins, 1);
                symbols = exp(1j*(pi/4 + qpskData*pi/2));
                X(idxStart:idxEnd) = symbols;
                x = ifft(ifftshift(X));
                xWithPrefix = [x(end-cyclicPrefixLength+1:end); x];
                sig = [sig; xWithPrefix]; %#ok<AGROW>
            end

        case 7
            sig = complex(zeros(Ntarget,1));
            numBursts = randi([2 8]);

            for b = 1:numBursts
                burstLength = randi([round(0.02*Ntarget) round(0.12*Ntarget)]);
                startIndex = randi([1 Ntarget-burstLength+1]);
                samplesPerSymbol = randi([6 20]);
                numSymbols = ceil(burstLength/samplesPerSymbol) + 10;
                data = randi([0 3], numSymbols, 1);
                symbols = exp(1j*(pi/4 + data*pi/2));
                burst = repelem(symbols, samplesPerSymbol);
                burst = repeatOrTrim(burst, burstLength);
                indices = startIndex:startIndex+burstLength-1;
                sig(indices) = sig(indices) + burst;
            end
    end

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateReceiverNoiseOnly(Fs, Ntarget)
    sig = complex(randn(Ntarget,1), randn(Ntarget,1));

    if rand < 0.80
        filterLength = randi([3 25]);
        h = rand(filterLength,1);
        h = h / sum(h);
        sig = filter(h, 1, sig);
    end

    if rand < 0.35
        t = (0:Ntarget-1).' / Fs;
        spurFrequency = (-18 + 36*rand)*1e6;
        sig = sig + 0.03*exp(1j*2*pi*spurFrequency*t);
    end

    sig = normalizeSignal(sig);
end

function [rx, info] = simulateReceiver(tx, Fs, snrDb)
    tx = normalizeSignal(tx);
    rx = tx * exp(1j*2*pi*rand);

    fineOffsetHz = (-1.2 + 2.4*rand)*1e6;
    rx = frequencyShift(rx, Fs, fineOffsetHz);

    if rand < 0.75
        numTaps = randi([2 12]);
        decay = exp(-(0:numTaps-1).'/3);
        h = (randn(numTaps,1) + 1j*randn(numTaps,1)).*decay;
        h = h / norm(h);
        rx = filter(h, 1, rx);
    end

    if rand < 0.85
        rx = applyIQImbalance(rx);
    end

    if rand < 0.70
        dcOffset = (0.005 + 0.06*rand)*exp(1j*2*pi*rand);
        rx = rx + dcOffset;
    end

    rx = normalizeSignal(rx);
    rx = awgn(rx, snrDb, "measured");

    gainDb = -8 + 16*rand;
    [rx, clipRatio] = simulateADC(rx, gainDb);

    info.gainDb = gainDb;
    info.fineOffsetHz = fineOffsetHz;
    info.clipRatio = clipRatio;
end

function [rx, info] = simulateNoiseObservation(noiseIn, Fs)
    rx = normalizeSignal(noiseIn);

    if rand < 0.75
        rx = rx + (0.005 + 0.07*rand)*exp(1j*2*pi*rand);
    end

    fineOffsetHz = 0;
    if rand < 0.45
        t = (0:length(rx)-1).' / Fs;
        fineOffsetHz = 0.05e6*randn;
        rx = rx + 0.02*exp(1j*2*pi*fineOffsetHz*t);
    end

    gainDb = -10 + 14*rand;
    [rx, clipRatio] = simulateADC(rx, gainDb);

    info.gainDb = gainDb;
    info.fineOffsetHz = fineOffsetHz;
    info.clipRatio = clipRatio;
end

function [y, clipRatio] = simulateADC(x, gainDb)
    x = normalizeSignal(x);
    x = x * 10^(gainDb/20);

    inPhase = real(x);
    quadrature = imag(x);
    clipMask = abs(inPhase) > 1 | abs(quadrature) > 1;
    clipRatio = mean(clipMask);

    inPhase = max(min(inPhase, 1), -1);
    quadrature = max(min(quadrature, 1), -1);

    inPhaseQuantized = int16(round(inPhase * 32767));
    quadratureQuantized = int16(round(quadrature * 32767));

    inPhaseRecovered = double(inPhaseQuantized) / 32768;
    quadratureRecovered = double(quadratureQuantized) / 32768;

    y = single(complex(inPhaseRecovered, quadratureRecovered));
    y = y(:);
end

function y = applyIQImbalance(x)
    amplitudeImbalanceDb = -4.5 + 9*rand;
    amplitudeScale = 10^(amplitudeImbalanceDb/20);
    phaseImbalance = (-5.5 + 11*rand) * pi/180;

    inPhase = real(x);
    quadrature = imag(x);
    inPhaseAdjusted = amplitudeScale * inPhase;
    quadratureAdjusted = (1/amplitudeScale) * quadrature;

    y = inPhaseAdjusted + 1j*(quadratureAdjusted*cos(phaseImbalance) + ...
        inPhaseAdjusted*sin(phaseImbalance));
end

function y = applyBurstMask(x)
    numSamples = length(x);
    mask = zeros(numSamples,1);
    numBursts = randi([1 5]);

    for b = 1:numBursts
        burstLength = randi([round(0.08*numSamples) round(0.45*numSamples)]);
        startIndex = randi([1 numSamples-burstLength+1]);
        fade = hann(min(128, burstLength));
        burstMask = ones(burstLength,1);
        halfLength = floor(length(fade)/2);

        if burstLength > length(fade) && halfLength > 0
            burstMask(1:halfLength) = fade(1:halfLength);
            burstMask(end-halfLength+1:end) = fade(end-halfLength+1:end);
        end

        indices = startIndex:startIndex+burstLength-1;
        mask(indices) = max(mask(indices), burstMask);
    end

    y = normalizeSignal(x(:).*mask);
end

function y = frequencyShift(x, Fs, frequencyOffset)
    t = (0:length(x)-1).' / Fs;
    y = x(:).*exp(1j*2*pi*frequencyOffset*t);
end

function snrDb = randomSNR()
    values = [-6 -4 -2 0 2 4 6 8 10 12 15 18 20 25 30];
    snrDb = values(randi(numel(values)));
end

function powerScale = randomRelativePower()
    values = [0.08 0.12 0.15 0.20 0.30 0.40 0.55 0.70 0.90 1.10 1.30 1.50];
    powerScale = values(randi(numel(values)));
end

function img = iqToSpectrogramImage(sig, Fs, imageSize)
    sig = normalizeSignal(sig(:));
    window = hamming(1024);
    overlapLength = 768;
    nfft = 2048;

    [spectrogramData,~,~] = spectrogram(sig, window, overlapLength, ...
        nfft, Fs, "centered");

    img = 20*log10(abs(spectrogramData) + eps);
    img = img - max(img(:));
    img = max(img, -90);
    img = (img + 90) / 90;
    img = imresize(img, imageSize);
end

function y = repeatOrTrim(x, targetLength)
    x = x(:);
    if length(x) < targetLength
        repetitions = ceil(targetLength/length(x));
        x = repmat(x, repetitions, 1);
    end
    y = x(1:targetLength);
end

function y = normalizeSignal(x)
    y = x ./ max(abs(x) + eps);
end

function valueMHz = toMHz(valueHz)
    if isnan(valueHz)
        valueMHz = NaN;
    else
        valueMHz = valueHz/1e6;
    end
end

function pathOut = relativePath(projectDir, pathIn)
    if strlength(string(pathIn)) == 0
        pathOut = "";
        return;
    end

    projectPrefix = string(projectDir) + filesep;
    pathOut = erase(string(pathIn), projectPrefix);
    pathOut = replace(pathOut, filesep, "/");
end
