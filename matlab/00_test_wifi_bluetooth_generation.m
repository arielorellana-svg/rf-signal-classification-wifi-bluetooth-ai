clear; clc; close all;

%% Parámetros generales
FsTarget = 20e6;        % Frecuencia común de muestreo para comparar señales
Ntarget = 40960;        % Longitud final de cada señal

%% =========================
%  1. Generar señal WiFi
%  =========================

cfgWiFi = wlanNonHTConfig;
cfgWiFi.ChannelBandwidth = "CBW20";
cfgWiFi.MCS = 4;
cfgWiFi.PSDULength = 1000;

bitsWiFi = randi([0 1], cfgWiFi.PSDULength*8, 1);

wifiSig = wlanWaveformGenerator(bitsWiFi, cfgWiFi);

FsWiFi = wlanSampleRate(cfgWiFi);

wifiSig = resample(wifiSig, FsTarget, FsWiFi);
wifiSig = repeatOrTrim(wifiSig, Ntarget);
wifiSig = normalizeSignal(wifiSig);

%% =========================
%  2. Generar señal Bluetooth LE
%  =========================

sps = 8;                % Samples per symbol
FsBT = 1e6 * sps;       % BLE LE1M usa 1 Msym/s

messageBT = randi([0 1], 2000, 1);

btSig = bleWaveformGenerator(messageBT, ...
    "Mode", "LE1M", ...
    "SamplesPerSymbol", sps);

btSig = resample(btSig, FsTarget, FsBT);
btSig = repeatOrTrim(btSig, Ntarget);
btSig = normalizeSignal(btSig);

%% =========================
%  3. Señal combinada WiFi + Bluetooth
%  =========================

t = (0:Ntarget-1).' / FsTarget;

% Mover Bluetooth +5 MHz para que se vea separado dentro del espectro
freqOffsetBT = 5e6;
btShifted = btSig .* exp(1j*2*pi*freqOffsetBT*t);

mixedSig = normalizeSignal(wifiSig + 0.5*btShifted);

%% =========================
%  4. Visualizar espectrogramas
%  =========================

figure;
plotSpectrogram(wifiSig, FsTarget, "Espectrograma WiFi");

figure;
plotSpectrogram(btShifted, FsTarget, "Espectrograma Bluetooth desplazado +5 MHz");

figure;
plotSpectrogram(mixedSig, FsTarget, "Espectrograma WiFi + Bluetooth");

disp("Prueba finalizada correctamente.");

%% =========================
%  Funciones auxiliares
%  =========================

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

function plotSpectrogram(sig, Fs, plotTitle)
    win = hamming(512);
    noverlap = 384;
    nfft = 1024;

    spectrogram(sig, win, noverlap, nfft, Fs, "centered", "yaxis");
    title(plotTitle);
end