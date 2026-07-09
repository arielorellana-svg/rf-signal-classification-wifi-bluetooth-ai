clear; clc; close all;

%% ============================================================
%  PRUEBA FINAL INDEPENDIENTE V2
%  WiFi / Bluetooth / Coexistencia / Noise / Unknown
%  Este dataset NO se usa para entrenar.
%% ============================================================

projectDir = pwd;

baseDir = fullfile(projectDir, "data", "blind_test_v2_final");

classes = ["WiFi", ...
           "Bluetooth", ...
           "WiFi_Bluetooth_Overlap", ...
           "WiFi_Bluetooth_Separated", ...
           "Noise", ...
           "Unknown"];

numImagesPerClass = 1000;   % 6 clases x 1000 = 6000 imágenes finales
FsTarget = 40e6;
Ntarget = 81920;
imageSize = [224 224];

resetDataset = true;
rng(2026);                  % Semilla diferente a entrenamiento y blind_test_v1

%% Crear carpetas
if resetDataset && exist(baseDir, "dir")
    fprintf("Eliminando prueba final anterior: %s\n", baseDir);
    rmdir(baseDir, "s");
end

for c = classes
    folder = fullfile(baseDir, c);
    if ~exist(folder, "dir")
        mkdir(folder);
    end
end

%% Metadata
totalImages = numImagesPerClass * numel(classes);

metaFile = strings(totalImages,1);
metaClass = strings(totalImages,1);
metaSNR = nan(totalImages,1);
metaBTOffsetMHz = nan(totalImages,1);
metaUnknownOffsetMHz = nan(totalImages,1);
metaScenario = strings(totalImages,1);

row = 0;

fprintf("\nGenerando prueba final independiente...\n");
fprintf("Carpeta: %s\n", baseDir);
fprintf("Imágenes por clase: %d\n", numImagesPerClass);
fprintf("Total esperado: %d imágenes\n\n", totalImages);

%% ============================================================
%  Bucle principal
%% ============================================================

for n = 1:numImagesPerClass

    %% 1. WiFi solo
    snrDb = randomSNRFinal();

    sig = generateWiFiSignal(FsTarget, Ntarget);
    sig = applyFinalRFImpairments(sig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi", sprintf("wifi_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi";
    metaSNR(row) = snrDb;
    metaScenario(row) = "WiFi solo final";

    %% 2. Bluetooth solo
    snrDb = randomSNRFinal();

    sig = generateBluetoothSignal(FsTarget, Ntarget);

    btOffset = (-17.5 + 35*rand) * 1e6;   % Offset continuo
    sig = frequencyShift(sig, FsTarget, btOffset);

    sig = applyFinalRFImpairments(sig, FsTarget, snrDb);

    filename = fullfile(baseDir, "Bluetooth", sprintf("bluetooth_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Bluetooth";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "Bluetooth solo final";

    %% 3. WiFi + Bluetooth solapados
    snrDb = randomSNRFinal();

    wifiSig = generateWiFiSignal(FsTarget, Ntarget);
    btSig = generateBluetoothSignal(FsTarget, Ntarget);

    btOffset = (-8.5 + 17*rand) * 1e6;    % Dentro o cerca del canal WiFi
    btSig = frequencyShift(btSig, FsTarget, btOffset);

    btPower = randomRelativePowerFinal();

    sig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);
    sig = applyFinalRFImpairments(sig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi_Bluetooth_Overlap", ...
        sprintf("wifi_bt_overlap_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi_Bluetooth_Overlap";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "WiFi + Bluetooth solapados final";

    %% 4. WiFi + Bluetooth separados
    snrDb = randomSNRFinal();

    wifiSig = generateWiFiSignal(FsTarget, Ntarget);
    btSig = generateBluetoothSignal(FsTarget, Ntarget);

    if rand < 0.5
        btOffset = (-19.5 + 5.5*rand) * 1e6;   % -19.5 a -14 MHz
    else
        btOffset = (14 + 5.5*rand) * 1e6;      % +14 a +19.5 MHz
    end

    btSig = frequencyShift(btSig, FsTarget, btOffset);

    btPower = randomRelativePowerFinal();

    sig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);
    sig = applyFinalRFImpairments(sig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi_Bluetooth_Separated", ...
        sprintf("wifi_bt_separated_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi_Bluetooth_Separated";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "WiFi + Bluetooth separados final";

    %% 5. Noise
    sig = generateFinalNoise(FsTarget, Ntarget);

    filename = fullfile(baseDir, "Noise", sprintf("noise_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Noise";
    metaScenario(row) = "Ruido final complejo";

    %% 6. Unknown
    snrDb = randomSNRFinal();

    sig = generateFinalUnknownSignal(FsTarget, Ntarget);

    unknownOffset = (-18.5 + 37*rand) * 1e6;
    sig = frequencyShift(sig, FsTarget, unknownOffset);

    sig = applyFinalRFImpairments(sig, FsTarget, snrDb);

    filename = fullfile(baseDir, "Unknown", sprintf("unknown_final_%05d.png", n));
    saveSpectrogramImage(sig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Unknown";
    metaSNR(row) = snrDb;
    metaUnknownOffsetMHz(row) = unknownOffset/1e6;
    metaScenario(row) = "Unknown final";

    %% Progreso
    if mod(n,50) == 0 || n == 1
        fprintf("Progreso: %d / %d por clase\n", n, numImagesPerClass);
    end
end

%% Guardar metadata
metaFile = metaFile(1:row);
metaClass = metaClass(1:row);
metaSNR = metaSNR(1:row);
metaBTOffsetMHz = metaBTOffsetMHz(1:row);
metaUnknownOffsetMHz = metaUnknownOffsetMHz(1:row);
metaScenario = metaScenario(1:row);

metadata = table( ...
    metaFile(:), ...
    metaClass(:), ...
    metaSNR(:), ...
    metaBTOffsetMHz(:), ...
    metaUnknownOffsetMHz(:), ...
    metaScenario(:));

metadata.Properties.VariableNames = { ...
    'File', ...
    'Class', ...
    'SNR_dB', ...
    'BT_Offset_MHz', ...
    'Unknown_Offset_MHz', ...
    'Scenario'};

metadataPath = fullfile(baseDir, "metadata_blind_test_v2_final.csv");
writetable(metadata, metadataPath);

fprintf("\nPrueba final independiente generada correctamente.\n");
fprintf("Metadata guardada en:\n%s\n", metadataPath);

%% Verificación
imdsFinal = imageDatastore(baseDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Conteo por clase:");
disp(countEachLabel(imdsFinal));

figure;
idx = randperm(numel(imdsFinal.Files), min(30,numel(imdsFinal.Files)));
montage(imdsFinal.Files(idx));
title("Muestras aleatorias - Prueba final independiente");

%% ============================================================
%  FUNCIONES AUXILIARES
%% ============================================================

function sig = generateWiFiSignal(FsTarget, Ntarget)
    cfgWiFi = wlanNonHTConfig;
    cfgWiFi.ChannelBandwidth = "CBW20";
    cfgWiFi.MCS = randi([0 7]);
    cfgWiFi.PSDULength = randi([150 2200]);

    bitsWiFi = randi([0 1], cfgWiFi.PSDULength*8, 1);

    sig = wlanWaveformGenerator(bitsWiFi, cfgWiFi);

    FsWiFi = wlanSampleRate(cfgWiFi);
    sig = resample(sig, FsTarget, FsWiFi);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateBluetoothSignal(FsTarget, Ntarget)
    sps = 8;
    FsBT = 1e6 * sps;

    msgLen = randi([220 2080]);
    messageBT = randi([0 1], msgLen, 1);

    sig = bleWaveformGenerator(messageBT, ...
        "Mode", "LE1M", ...
        "SamplesPerSymbol", sps);

    sig = resample(sig, FsTarget, FsBT);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateFinalUnknownSignal(FsTarget, Ntarget)
    signalType = randi([1 7]);

    switch signalType
        case 1
            % BPSK angosta
            sps = randi([8 32]);
            numSymbols = ceil(Ntarget/sps) + 100;
            bits = randi([0 1], numSymbols, 1);
            symbols = 2*bits - 1;
            sig = repelem(symbols, sps);
            sig = complex(sig, zeros(size(sig)));

            h = hamming(41);
            h = h / sum(h);
            sig = filter(h, 1, sig);

        case 2
            % QPSK
            sps = randi([6 26]);
            numSymbols = ceil(Ntarget/sps) + 100;
            data = randi([0 3], numSymbols, 1);
            symbols = exp(1j*(pi/4 + data*pi/2));
            sig = repelem(symbols, sps);

            h = hamming(31);
            h = h / sum(h);
            sig = filter(h, 1, sig);

        case 3
            % Chirp complejo
            t = (0:Ntarget-1).' / FsTarget;
            f0 = (-7 + 14*rand)*1e6;
            f1 = (-7 + 14*rand)*1e6;
            k = (f1 - f0) / max(t);

            phase = 2*pi*(f0*t + 0.5*k*t.^2);
            sig = exp(1j*phase);

        case 4
            % Multitono
            t = (0:Ntarget-1).' / FsTarget;
            sig = zeros(Ntarget,1);
            numTones = randi([2 6]);

            for k = 1:numTones
                f = (-7 + 14*rand)*1e6;
                amp = 0.20 + 0.90*rand;
                sig = sig + amp*exp(1j*2*pi*f*t);
            end

        case 5
            % FSK sintético
            sps = randi([6 22]);
            numSymbols = ceil(Ntarget/sps) + 100;
            bits = randi([0 1], numSymbols, 1);
            freqDev = (0.20 + 1.60*rand)*1e6;

            freqSeq = (2*bits - 1)*freqDev;
            freqSamples = repelem(freqSeq, sps);
            freqSamples = repeatOrTrim(freqSamples, Ntarget);

            phase = cumsum(2*pi*freqSamples/FsTarget);
            sig = exp(1j*phase);

        case 6
            % OFDM sintética no WiFi
            nfft = 256;
            cpLen = 32;
            numSym = ceil(Ntarget/(nfft+cpLen)) + 10;

            activeBins = 30 + randi([0 110]);
            sig = [];

            for k = 1:numSym
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
            % Señal tipo ráfaga pulsada
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

function sig = generateFinalNoise(FsTarget, Ntarget)
    sig = complex(randn(Ntarget,1), randn(Ntarget,1));

    % Ruido coloreado
    if rand < 0.8
        filtLen = randi([3 25]);
        h = rand(filtLen,1);
        h = h / sum(h);
        sig = filter(h, 1, sig);
    end

    % DC offset
    if rand < 0.6
        sig = sig + (0.005 + 0.07*rand)*exp(1j*2*pi*rand);
    end

    % Espurio débil
    if rand < 0.3
        t = (0:Ntarget-1).' / FsTarget;
        fspur = (-18 + 36*rand)*1e6;
        sig = sig + 0.04*exp(1j*2*pi*fspur*t);
    end

    % Ráfaga de ruido
    if rand < 0.3
        burstLen = randi([round(0.05*Ntarget) round(0.25*Ntarget)]);
        startIdx = randi([1 Ntarget-burstLen+1]);
        burst = 0.15 * complex(randn(burstLen,1), randn(burstLen,1));
        sig(startIdx:startIdx+burstLen-1) = sig(startIdx:startIdx+burstLen-1) + burst;
    end

    sig = normalizeSignal(sig);
end

function sig = applyFinalRFImpairments(sig, Fs, snrDb)
    sig = normalizeSignal(sig);

    % Fase aleatoria
    sig = sig * exp(1j*2*pi*rand);

    % Offset fino continuo
    if rand < 0.95
        fineOffset = (-1.3 + 2.6*rand) * 1e6;
        sig = frequencyShift(sig, Fs, fineOffset);
    end

    % Desplazamiento temporal
    if rand < 0.75
        maxShift = round(0.055 * length(sig));
        shift = randi([-maxShift maxShift]);

        if shift > 0
            sig = [zeros(shift,1); sig(1:end-shift)];
        elseif shift < 0
            shift = abs(shift);
            sig = [sig(shift+1:end); zeros(shift,1)];
        end
    end

    % Multipath
    if rand < 0.8
        nTaps = randi([2 14]);
        decay = exp(-(0:nTaps-1).'/3);
        h = (randn(nTaps,1) + 1j*randn(nTaps,1)) .* decay;
        h = h / norm(h);
        sig = filter(h, 1, sig);
    end

    % IQ imbalance
    if rand < 0.85
        sig = applyIQImbalance(sig);
    end

    % DC offset
    if rand < 0.65
        dc = (0.005 + 0.055*rand) * exp(1j*2*pi*rand);
        sig = sig + dc;
    end

    sig = normalizeSignal(sig);

    % AWGN
    sig = awgn(sig, snrDb, "measured");

    % Variación de amplitud
    sig = sig * (0.30 + 1.50*rand);

    sig = normalizeSignal(sig);
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

function y = frequencyShift(x, Fs, freqOffset)
    t = (0:length(x)-1).' / Fs;
    y = x .* exp(1j*2*pi*freqOffset*t);
end

function snrDb = randomSNRFinal()
    snrList = [-9 -7 -5 -3 -1 0 2 4 6 8 10 12 15 18 20 25 30];
    snrDb = snrList(randi(numel(snrList)));
end

function p = randomRelativePowerFinal()
    powerList = [0.06 0.10 0.15 0.20 0.30 0.40 0.55 0.70 0.90 1.10 1.30 1.60];
    p = powerList(randi(numel(powerList)));
end

function saveSpectrogramImage(sig, Fs, filename, imageSize)
    win = hamming(1024);
    noverlap = 768;
    nfft = 2048;

    [s,~,~] = spectrogram(sig, win, noverlap, nfft, Fs, "centered");

    img = 20*log10(abs(s) + eps);

    img = img - max(img(:));
    img = max(img, -90);
    img = (img + 90) / 90;

    img = imresize(img, imageSize);

    imwrite(img, filename);
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
