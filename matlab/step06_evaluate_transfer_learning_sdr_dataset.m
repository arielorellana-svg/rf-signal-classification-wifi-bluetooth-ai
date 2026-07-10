clear; clc; close all;

%% ============================================================
%  STEP 06 - EVALUATE TRANSFER LEARNING MODEL ON SDR IMAGE DATASET
%  Model: ResNet-18 transfer learning
%  Dataset: data_sdr
%
%  This script evaluates the transfer-learning model using spectrogram
%  images generated from SDR captures.
%% ============================================================

%% Project paths
projectDir = "C:\Users\DETPC\Desktop\Proyecto";

modelPath = fullfile(projectDir, "models", ...
    "resnet18_transfer_learning_wifi_bluetooth.mat");

sdrRoot = fullfile(projectDir, "data_sdr");

resultsDir = fullfile(projectDir, "results");

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end

if ~exist(sdrRoot, "dir")
    error("SDR dataset folder not found: %s", sdrRoot);
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

%% Detect SDR spectrogram folder
% Supports either:
%   data_sdr/ClassName/*.png
% or:
%   data_sdr/spectrograms/ClassName/*.png

directCandidate = sdrRoot;
nestedCandidate = fullfile(sdrRoot, "spectrograms");

expectedClasses = ["Bluetooth", ...
                   "Noise", ...
                   "Unknown", ...
                   "WiFi", ...
                   "WiFi_Bluetooth_Overlap", ...
                   "WiFi_Bluetooth_Separated"];

hasDirectClasses = all(arrayfun(@(c) exist(fullfile(directCandidate, c), "dir") == 7, expectedClasses));
hasNestedClasses = all(arrayfun(@(c) exist(fullfile(nestedCandidate, c), "dir") == 7, expectedClasses));

if hasDirectClasses
    sdrDir = directCandidate;
elseif hasNestedClasses
    sdrDir = nestedCandidate;
else
    fprintf("\nCould not find all expected class folders directly or inside data_sdr/spectrograms.\n");
    fprintf("Listing data_sdr contents:\n");
    disp(dir(sdrRoot));
    error("Please check the SDR dataset folder structure.");
end

fprintf("Using SDR spectrogram folder:\n%s\n", sdrDir);

%% Load trained transfer-learning model
S = load(modelPath);

if isfield(S, "netTL")
    netTL = S.netTL;
else
    error("The file does not contain variable 'netTL': %s", modelPath);
end

if isfield(S, "classNames")
    classNames = S.classNames;
else
    classNames = cellstr(expectedClasses);
end

if isfield(S, "inputSize")
    inputSize = S.inputSize;
else
    inputSize = [224 224 3];
end

fprintf("\nLoaded transfer-learning model:\n%s\n", modelPath);
fprintf("Input size: %d x %d x %d\n", inputSize(1), inputSize(2), inputSize(3));

%% Load SDR image dataset
imdsSDR = imageDatastore(sdrDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("SDR dataset count by class:");
disp(countEachLabel(imdsSDR));

sdrClassNames = categories(imdsSDR.Labels);

disp("Model classes:");
disp(classNames);

disp("SDR dataset classes:");
disp(sdrClassNames);

if ~isequal(string(classNames(:)), string(sdrClassNames(:)))
    warning("Class names or class order differ. Metrics will use the model class order.");
end

%% Preview first SDR image
img = readimage(imdsSDR, 1);
fprintf("\nFirst SDR image size:\n");
disp(size(img));

%% Prepare grayscale images for ResNet-18 RGB input
augimdsSDR = augmentedImageDatastore(inputSize(1:2), imdsSDR, ...
    "ColorPreprocessing", "gray2rgb");

%% Classify SDR dataset
miniBatchSize = 64;

fprintf("\nClassifying SDR image dataset...\n");

YPredSDR = classify(netTL, augimdsSDR, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");

YTrueSDR = imdsSDR.Labels;

sdrAccuracyTL = mean(YPredSDR == YTrueSDR);

fprintf("\nTransfer learning SDR dataset accuracy: %.2f %%\n", ...
    sdrAccuracyTL * 100);

%% Confusion matrix and metrics
% confusionmat supports fixed class order using 'Order'.
classOrder = categorical(classNames, classNames);
C = confusionmat(YTrueSDR, YPredSDR, 'Order', classOrder);

fig = figure;
cm = confusionchart(C, categorical(classNames, classNames));
cm.Title = "Confusion Matrix - ResNet-18 Transfer Learning SDR Dataset";
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

metricsTableSDRTL = table( ...
    string(classNames(:)), ...
    precision(:), ...
    recall(:), ...
    f1score(:), ...
    'VariableNames', {'Class','Precision','Recall','F1_score'});

disp("Transfer learning metrics on SDR dataset:");
disp(metricsTableSDRTL);

summarySDRTL = table( ...
    string("resnet18_transfer_learning"), ...
    numel(classNames), ...
    numel(imdsSDR.Files), ...
    sdrAccuracyTL, ...
    miniBatchSize, ...
    string(sdrDir), ...
    'VariableNames', { ...
        'Model', ...
        'NumClasses', ...
        'NumSDRSamples', ...
        'SDRAccuracy', ...
        'MiniBatchSize', ...
        'SDRDatasetPath'});

disp("SDR evaluation summary:");
disp(summarySDRTL);

%% Save results
resultsPath = fullfile(resultsDir, ...
    "sdr_results_transfer_learning.mat");

metricsCsvPath = fullfile(resultsDir, ...
    "metrics_transfer_learning_sdr_dataset.csv");

summaryCsvPath = fullfile(resultsDir, ...
    "summary_transfer_learning_sdr_dataset.csv");

confusionFigPath = fullfile(resultsDir, ...
    "confusion_matrix_transfer_learning_sdr_dataset.png");

save(resultsPath, ...
    "sdrAccuracyTL", ...
    "metricsTableSDRTL", ...
    "summarySDRTL", ...
    "C", ...
    "YTrueSDR", ...
    "YPredSDR", ...
    "classNames", ...
    "sdrDir", ...
    "-v7.3");

writetable(metricsTableSDRTL, metricsCsvPath);
writetable(summarySDRTL, summaryCsvPath);
exportgraphics(fig, confusionFigPath, "Resolution", 300);

fprintf("\nSaved SDR transfer-learning results:\n%s\n", resultsPath);
fprintf("\nSaved SDR metrics:\n%s\n", metricsCsvPath);
fprintf("\nSaved SDR summary:\n%s\n", summaryCsvPath);
fprintf("\nSaved SDR confusion matrix:\n%s\n", confusionFigPath);

fprintf("\nTransfer-learning SDR evaluation completed successfully.\n");