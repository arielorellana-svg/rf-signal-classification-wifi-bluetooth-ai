clear; clc; close all;

%% ============================================================
%  DATASET V3 DOMAIN-RANDOMIZED
%  Clasificación RF: WiFi / Bluetooth / Coexistencia / Noise / Unknown
%  Proyecto: Clasificación de señales RF usando Deep Learning
%  Autor: Ariel Orellana
%% ============================================================

%% Configuración principal
projectDir = pwd;

baseDir = fullfile(projectDir, "data", "spectrograms_v3_domain_randomized");

classes = ["WiFi", ...
           "Bluetooth", ...
           "WiFi_Bluetooth_Overlap", ...
           "WiFi_Bluetooth_Separated", ...
           "Noise", ...
           "Unknown"];

numImagesPerClass = 3000;   % 6 clases x 3000 = 18000 imágenes
FsTarget = 40e6;            % Banda observada: -20 MHz a +20 MHz
Ntarget = 81920;            % Duración aproximada: 2.048 ms
imageSize = [224 224];      % Entrada para CNN

resetDataset = true;        % true: elimina y regenera el dataset v3

rng(21);                    % Reproducibilidad

%% Crear carpetas
if resetDataset && exist(baseDir, "dir")
    fprintf("Eliminando dataset anterior: %s\n", baseDir);
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

fprintf("\nGenerando dataset V3 domain-randomized...\n");
fprintf("Carpeta: %s\n", baseDir);
fprintf("Clases: %d\n", numel(classes));
fprintf("Imágenes por clase: %d\n", numImagesPerClass);
fprintf("Total esperado: %d imágenes\n\n", totalImages);

%% ============================================================
%  Bucle principal
%% ============================================================

for n = 1:numImagesPerClass

    %% -------------------------
    % 1. WiFi
    %% -------------------------
    snrDb = randomSNR();

    wifiSig = generateWiFiSignal(FsTarget, Ntarget);
    wifiSig = applyRFImpairments(wifiSig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi", sprintf("wifi_%05d.png", n));
    saveSpectrogramImage(wifiSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi";
    metaSNR(row) = snrDb;
    metaScenario(row) = "WiFi solo";

    %% -------------------------
    % 2. Bluetooth
    %% -------------------------
    snrDb = randomSNR();

    btSig = generateBluetoothSignal(FsTarget, Ntarget);

    % Offset continuo: evita que la red aprenda solo posiciones fijas
    btOffset = (-17 + 34*rand) * 1e6;    % -17 a +17 MHz
    btSig = frequencyShift(btSig, FsTarget, btOffset);

    btSig = applyRFImpairments(btSig, FsTarget, snrDb);

    filename = fullfile(baseDir, "Bluetooth", sprintf("bluetooth_%05d.png", n));
    saveSpectrogramImage(btSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Bluetooth";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "Bluetooth solo";

    %% -------------------------
    % 3. WiFi + Bluetooth solapados
    %% -------------------------
    snrDb = randomSNR();

    wifiSig = generateWiFiSignal(FsTarget, Ntarget);
    btSig = generateBluetoothSignal(FsTarget, Ntarget);

    % Bluetooth dentro o cerca del canal WiFi
    btOffset = (-8 + 16*rand) * 1e6;     % -8 a +8 MHz
    btSig = frequencyShift(btSig, FsTarget, btOffset);

    btPower = randomRelativePower();

    mixedSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);
    mixedSig = applyRFImpairments(mixedSig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi_Bluetooth_Overlap", ...
        sprintf("wifi_bt_overlap_%05d.png", n));
    saveSpectrogramImage(mixedSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi_Bluetooth_Overlap";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "WiFi + Bluetooth solapados";

    %% -------------------------
    % 4. WiFi + Bluetooth separados
    %% -------------------------
    snrDb = randomSNR();

    wifiSig = generateWiFiSignal(FsTarget, Ntarget);
    btSig = generateBluetoothSignal(FsTarget, Ntarget);

    % Bluetooth fuera del canal principal WiFi
    if rand < 0.5
        btOffset = (-19 + 5*rand) * 1e6;   % -19 a -14 MHz
    else
        btOffset = (14 + 5*rand) * 1e6;    % +14 a +19 MHz
    end

    btSig = frequencyShift(btSig, FsTarget, btOffset);

    btPower = randomRelativePower();

    mixedSig = normalizeSignal(wifiSig) + btPower*normalizeSignal(btSig);
    mixedSig = applyRFImpairments(mixedSig, FsTarget, snrDb);

    filename = fullfile(baseDir, "WiFi_Bluetooth_Separated", ...
        sprintf("wifi_bt_separated_%05d.png", n));
    saveSpectrogramImage(mixedSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "WiFi_Bluetooth_Separated";
    metaSNR(row) = snrDb;
    metaBTOffsetMHz(row) = btOffset/1e6;
    metaScenario(row) = "WiFi + Bluetooth separados";

    %% -------------------------
    % 5. Noise
    %% -------------------------
    noiseSig = generateHardNoise(FsTarget, Ntarget);

    filename = fullfile(baseDir, "Noise", sprintf("noise_%05d.png", n));
    saveSpectrogramImage(noiseSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Noise";
    metaScenario(row) = "Ruido complejo / coloreado";

    %% -------------------------
    % 6. Unknown
    %% -------------------------
    snrDb = randomSNR();

    unknownSig = generateUnknownSignal(FsTarget, Ntarget);

    unknownOffset = (-18 + 36*rand) * 1e6;     % -18 a +18 MHz
    unknownSig = frequencyShift(unknownSig, FsTarget, unknownOffset);

    unknownSig = applyRFImpairments(unknownSig, FsTarget, snrDb);

    filename = fullfile(baseDir, "Unknown", sprintf("unknown_%05d.png", n));
    saveSpectrogramImage(unknownSig, FsTarget, filename, imageSize);

    row = row + 1;
    metaFile(row) = string(filename);
    metaClass(row) = "Unknown";
    metaSNR(row) = snrDb;
    metaUnknownOffsetMHz(row) = unknownOffset/1e6;
    metaScenario(row) = "Señal desconocida sintética";

    %% Progreso
    if mod(n,50) == 0 || n == 1
        fprintf("Progreso: %d / %d por clase\n", n, numImagesPerClass);
    end
end

%% ============================================================
%  Guardar metadata
%% ============================================================

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

metadataPath = fullfile(baseDir, "metadata_dataset_v3.csv");
writetable(metadata, metadataPath);

fprintf("\nDataset V3 generado correctamente.\n");
fprintf("Metadata guardada en:\n%s\n", metadataPath);

%% Verificación rápida
imds = imageDatastore(baseDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Conteo por clase:");
disp(countEachLabel(imds));

figure;
idx = randperm(numel(imds.Files), min(30,numel(imds.Files)));
montage(imds.Files(idx));
title("Muestras aleatorias del dataset V3");

%% ============================================================
%  FUNCIONES AUXILIARES
%% ============================================================

function sig = generateWiFiSignal(FsTarget, Ntarget)
    cfgWiFi = wlanNonHTConfig;
    cfgWiFi.ChannelBandwidth = "CBW20";
    cfgWiFi.MCS = randi([0 7]);
    cfgWiFi.PSDULength = randi([200 2000]);

    bitsWiFi = randi([0 1], cfgWiFi.PSDULength*8, 1);

    sig = wlanWaveformGenerator(bitsWiFi, cfgWiFi);

    FsWiFi = wlanSampleRate(cfgWiFi);
    sig = resample(sig, FsTarget, FsWiFi);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateBluetoothSignal(FsTarget, Ntarget)
    sps = 8;
    FsBT = 1e6 * sps;   % BLE LE1M: 1 Msym/s con 8 muestras por símbolo

    % bleWaveformGenerator admite hasta 2088 bits.
    % Usamos 2080 para evitar errores.
    msgLen = randi([250 2080]);
    messageBT = randi([0 1], msgLen, 1);

    sig = bleWaveformGenerator(messageBT, ...
        "Mode", "LE1M", ...
        "SamplesPerSymbol", sps);

    sig = resample(sig, FsTarget, FsBT);

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateUnknownSignal(FsTarget, Ntarget)
    signalType = randi([1 6]);

    switch signalType
        case 1
            % BPSK angosta
            sps = randi([10 30]);
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
            sps = randi([8 24]);
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
            f0 = (-6 + 12*rand)*1e6;
            f1 = (-6 + 12*rand)*1e6;
            k = (f1 - f0) / max(t);

            phase = 2*pi*(f0*t + 0.5*k*t.^2);
            sig = exp(1j*phase);

        case 4
            % Multitono
            t = (0:Ntarget-1).' / FsTarget;
            sig = zeros(Ntarget,1);
            numTones = randi([2 5]);

            for k = 1:numTones
                f = (-6 + 12*rand)*1e6;
                amp = 0.25 + 0.75*rand;
                sig = sig + amp*exp(1j*2*pi*f*t);
            end

        case 5
            % FSK sintético
            sps = randi([8 20]);
            numSymbols = ceil(Ntarget/sps) + 100;
            bits = randi([0 1], numSymbols, 1);
            freqDev = (0.25 + 1.25*rand)*1e6;

            freqSeq = (2*bits - 1)*freqDev;
            freqSamples = repelem(freqSeq, sps);
            freqSamples = repeatOrTrim(freqSamples, Ntarget);

            phase = cumsum(2*pi*freqSamples/FsTarget);
            sig = exp(1j*phase);

        case 6
            % Señal OFDM sintética no WiFi
            nfft = 256;
            cpLen = 32;
            numSym = ceil(Ntarget/(nfft+cpLen)) + 10;

            activeBins = 40 + randi([0 80]);
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
    end

    sig = repeatOrTrim(sig, Ntarget);
    sig = normalizeSignal(sig);
end

function sig = generateHardNoise(FsTarget, Ntarget)
    sig = complex(randn(Ntarget,1), randn(Ntarget,1));

    % Ruido coloreado
    if rand < 0.75
        filtLen = randi([3 21]);
        h = rand(filtLen,1);
        h = h / sum(h);
        sig = filter(h, 1, sig);
    end

    % DC offset
    if rand < 0.55
        sig = sig + (0.005 + 0.06*rand)*exp(1j*2*pi*rand);
    end

    % Tono espurio débil
    if rand < 0.25
        t = (0:Ntarget-1).' / FsTarget;
        fspur = (-18 + 36*rand)*1e6;
        sig = sig + 0.05*exp(1j*2*pi*fspur*t);
    end

    % Ráfaga débil ocasional
    if rand < 0.25
        burstLen = randi([round(0.05*Ntarget) round(0.25*Ntarget)]);
        startIdx = randi([1 Ntarget-burstLen+1]);
        burst = 0.15 * complex(randn(burstLen,1), randn(burstLen,1));
        sig(startIdx:startIdx+burstLen-1) = sig(startIdx:startIdx+burstLen-1) + burst;
    end

    sig = normalizeSignal(sig);
end

function sig = applyRFImpairments(sig, Fs, snrDb)
    sig = normalizeSignal(sig);

    % Fase aleatoria
    sig = sig * exp(1j*2*pi*rand);

    % Offset fino continuo
    if rand < 0.9
        fineOffset = (-1.2 + 2.4*rand) * 1e6;
        sig = frequencyShift(sig, Fs, fineOffset);
    end

    % Desplazamiento temporal
    if rand < 0.7
        maxShift = round(0.05 * length(sig));
        shift = randi([-maxShift maxShift]);

        if shift > 0
            sig = [zeros(shift,1); sig(1:end-shift)];
        elseif shift < 0
            shift = abs(shift);
            sig = [sig(shift+1:end); zeros(shift,1)];
        end
    end

    % Multipath
    if rand < 0.75
        nTaps = randi([2 12]);
        decay = exp(-(0:nTaps-1).'/3);
        h = (randn(nTaps,1) + 1j*randn(nTaps,1)) .* decay;
        h = h / norm(h);
        sig = filter(h, 1, sig);
    end

    % IQ imbalance
    if rand < 0.8
        sig = applyIQImbalance(sig);
    end

    % DC offset
    if rand < 0.65
        dc = (0.005 + 0.05*rand) * exp(1j*2*pi*rand);
        sig = sig + dc;
    end

    sig = normalizeSignal(sig);

    % AWGN
    sig = awgn(sig, snrDb, "measured");

    % Variación de amplitud
    sig = sig * (0.35 + 1.4*rand);

    sig = normalizeSignal(sig);
end

function y = applyIQImbalance(x)
    ampImbDb = -4 + 8*rand;               % -4 a +4 dB
    amp = 10^(ampImbDb/20);

    phaseImb = (-5 + 10*rand) * pi/180;   % -5 a +5 grados

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

function snrDb = randomSNR()
    % Incluye SNR muy bajo, bajo, medio y alto
    snrList = [-8 -6 -4 -2 0 2 4 6 8 10 12 15 18 20 25 30];
    snrDb = snrList(randi(numel(snrList)));
end

function p = randomRelativePower()
    % Potencia relativa Bluetooth/WiFi
    % Incluye Bluetooth muy débil, medio y dominante
    powerList = [0.08 0.12 0.15 0.20 0.30 0.40 0.55 0.70 0.90 1.10 1.30 1.50];
    p = powerList(randi(numel(powerList)));
end

function saveSpectrogramImage(sig, Fs, filename, imageSize)
    win = hamming(1024);
    noverlap = 768;
    nfft = 2048;

    [s,~,~] = spectrogram(sig, win, noverlap, nfft, Fs, "centered");

    img = 20*log10(abs(s) + eps);

    % Normalización relativa controlada
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
