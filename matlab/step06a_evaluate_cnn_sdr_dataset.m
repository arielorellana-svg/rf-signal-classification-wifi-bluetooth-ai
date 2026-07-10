clear; clc; close all;

%% ============================================================
%  STEP 06A - EVALUATE CUSTOM CNN BASELINE ON SDR IMAGE DATASET
%  Model: Custom CNN baseline
%  Dataset: data_sdr/spectrograms
%
%  This script regenerates the SDR validation results for the original
%  custom CNN baseline without using the word "simulation".
%% ============================================================

%% Project paths
projectDir = "C:\Users\DETPC\Desktop\Proyecto";

modelPath = fullfile(projectDir, "models", ...
    "cnn_wifi_bluetooth_v3_domain_randomized.mat");

sdrDir = fullfile(projectDir, "data_sdr", "spectrograms");

resultsDir = fullfile(projectDir, "results");

if ~exist(modelPath, "file")
    error("Custom CNN model not found: %s", modelPath);
end

if ~exist(sdrDir, "dir")
    error("SDR spectrogram dataset folder not found: %s", sdrDir);
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

%% Load custom CNN model
S = load(modelPath);

if isfield(S, "net")
    net = S.net;
else
    error("The model file does not contain variable 'net': %s", modelPath);
end

if isfield(S, "classNames")
    classNames = S.classNames;
else
    classNames = ["Bluetooth"; ...
                  "Noise"; ...
                  "Unknown"; ...
                  "WiFi"; ...
                  "WiFi_Bluetooth_Overlap"; ...
                  "WiFi_Bluetooth_Separated"];
end

classNames = string(classNames(:));

fprintf("Loaded custom CNN baseline model:\n%s\n", modelPath);

%% Load SDR spectrogram dataset
imdsSDR = imageDatastore(sdrDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("SDR dataset count by class:");
disp(countEachLabel(imdsSDR));

img = readimage(imdsSDR, 1);
fprintf("First SDR image size:\n");
disp(size(img));

%% Evaluate model
miniBatchSize = 64;

fprintf("\nClassifying SDR image dataset with custom CNN baseline...\n");

YPredSDR = classify(net, imdsSDR, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");

YTrueSDR = imdsSDR.Labels;

sdrAccuracyCNN = mean(YPredSDR == YTrueSDR);

fprintf("\nCustom CNN SDR dataset accuracy: %.2f %%\n", sdrAccuracyCNN * 100);

%% Confusion matrix and metrics
classOrder = categorical(classNames, classNames);

C = confusionmat(YTrueSDR, YPredSDR, 'Order', classOrder);

fig = figure;
cm = confusionchart(C, categorical(classNames, classNames));
cm.Title = "Confusion Matrix - Custom CNN SDR Dataset";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

TP = diag(C);
FP = sum(C,1)' - TP;
FN = sum(C,2) - TP;

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
f1score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1score(isnan(f1score)) = 0;

metricsTableCNNSDR = table( ...
    string(classNames(:)), ...
    precision(:), ...
    recall(:), ...
    f1score(:), ...
    'VariableNames', {'Class','Precision','Recall','F1_score'});

disp("Custom CNN metrics on SDR dataset:");
disp(metricsTableCNNSDR);

summaryCNNSDR = table( ...
    string("custom_cnn_baseline"), ...
    numel(classNames), ...
    numel(imdsSDR.Files), ...
    sdrAccuracyCNN, ...
    miniBatchSize, ...
    string(sdrDir), ...
    'VariableNames', { ...
        'Model', ...
        'NumClasses', ...
        'NumSDRSamples', ...
        'SDRAccuracy', ...
        'MiniBatchSize', ...
        'SDRDatasetPath'});

disp("Custom CNN SDR evaluation summary:");
disp(summaryCNNSDR);

%% Save results with neutral names
resultsPath = fullfile(resultsDir, ...
    "sdr_results_cnn_baseline.mat");

metricsCsvPath = fullfile(resultsDir, ...
    "metrics_cnn_sdr_dataset.csv");

summaryCsvPath = fullfile(resultsDir, ...
    "summary_cnn_sdr_dataset.csv");

confusionFigPath = fullfile(resultsDir, ...
    "confusion_matrix_cnn_sdr_dataset.png");

save(resultsPath, ...
    "sdrAccuracyCNN", ...
    "metricsTableCNNSDR", ...
    "summaryCNNSDR", ...
    "C", ...
    "YTrueSDR", ...
    "YPredSDR", ...
    "classNames", ...
    "sdrDir", ...
    "-v7.3");

writetable(metricsTableCNNSDR, metricsCsvPath);
writetable(summaryCNNSDR, summaryCsvPath);
exportgraphics(fig, confusionFigPath, "Resolution", 300);

fprintf("\nSaved custom CNN SDR results:\n%s\n", resultsPath);
fprintf("\nSaved custom CNN SDR metrics:\n%s\n", metricsCsvPath);
fprintf("\nSaved custom CNN SDR summary:\n%s\n", summaryCsvPath);
fprintf("\nSaved custom CNN SDR confusion matrix:\n%s\n", confusionFigPath);

fprintf("\nCustom CNN SDR evaluation completed successfully.\n");