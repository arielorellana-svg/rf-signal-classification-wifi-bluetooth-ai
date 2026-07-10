clear; clc; close all;

%% ============================================================
%  STEP 05B - EVALUATE TRANSFER LEARNING MODEL ON FINAL BLIND TEST
%  Model: ResNet-18 transfer learning
%  Dataset: blind_test_v2_final
%
%  This script evaluates the transfer-learning model using the same final
%  independent blind-test dataset used to evaluate the custom CNN baseline.
%% ============================================================

%% Project paths
projectDir = "C:\Users\DETPC\Desktop\Proyecto";

modelPath = fullfile(projectDir, "models", ...
    "resnet18_transfer_learning_wifi_bluetooth.mat");

blindDir = fullfile(projectDir, "data", "blind_test_v2_final");

resultsDir = fullfile(projectDir, "results");

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end

if ~exist(blindDir, "dir")
    error("Blind-test dataset folder not found: %s", blindDir);
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

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
    error("The file does not contain variable 'classNames': %s", modelPath);
end

if isfield(S, "inputSize")
    inputSize = S.inputSize;
else
    inputSize = [224 224 3];
end

fprintf("Loaded transfer-learning model:\n%s\n", modelPath);
fprintf("Input size: %d x %d x %d\n", inputSize(1), inputSize(2), inputSize(3));

%% Load final blind-test dataset
imdsBlind = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Blind-test dataset count by class:");
disp(countEachLabel(imdsBlind));

%% Check class consistency
blindClassNames = categories(imdsBlind.Labels);

disp("Model classes:");
disp(classNames);

disp("Blind-test classes:");
disp(blindClassNames);

if ~isequal(string(classNames(:)), string(blindClassNames(:)))
    warning("Class order or class names differ between model and blind-test dataset. Metrics will use the model class order.");
end

%% Prepare grayscale spectrograms for ResNet-18 RGB input
% The spectrogram images are grayscale, while ResNet-18 expects RGB input.
% augmentedImageDatastore is used for this preprocessing step.
augimdsBlind = augmentedImageDatastore(inputSize(1:2), imdsBlind, ...
    "ColorPreprocessing", "gray2rgb");

%% Classification
miniBatchSize = 64;

fprintf("\nClassifying final blind-test dataset...\n");

YPredBlind = classify(netTL, augimdsBlind, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");

YTrueBlind = imdsBlind.Labels;

finalBlindAccuracyTL = mean(YPredBlind == YTrueBlind);

fprintf("\nTransfer learning final blind-test accuracy: %.2f %%\n", ...
    finalBlindAccuracyTL * 100);

%% Confusion matrix and metrics
classOrder = categorical(classNames, classNames);

C = confusionmat(YTrueBlind, YPredBlind, 'Order', classOrder);

fig = figure;
cm = confusionchart(C, categorical(classNames, classNames));
cm.Title = "Confusion Matrix - ResNet-18 Transfer Learning Final Blind Test";
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

metricsTableBlindTL = table( ...
    string(classNames(:)), ...
    precision(:), ...
    recall(:), ...
    f1score(:), ...
    'VariableNames', {'Class','Precision','Recall','F1_score'});

disp("Transfer learning metrics on final blind test:");
disp(metricsTableBlindTL);

summaryBlindTL = table( ...
    string("resnet18_transfer_learning"), ...
    numel(classNames), ...
    numel(imdsBlind.Files), ...
    finalBlindAccuracyTL, ...
    miniBatchSize, ...
    'VariableNames', { ...
        'Model', ...
        'NumClasses', ...
        'NumBlindTestSamples', ...
        'FinalBlindTestAccuracy', ...
        'MiniBatchSize'});

disp("Final blind-test summary:");
disp(summaryBlindTL);

%% Save results
resultsPath = fullfile(resultsDir, ...
    "blind_test_results_transfer_learning.mat");

metricsCsvPath = fullfile(resultsDir, ...
    "metrics_transfer_learning_blind_test.csv");

summaryCsvPath = fullfile(resultsDir, ...
    "summary_transfer_learning_blind_test.csv");

confusionFigPath = fullfile(resultsDir, ...
    "confusion_matrix_transfer_learning_blind_test.png");

save(resultsPath, ...
    "finalBlindAccuracyTL", ...
    "metricsTableBlindTL", ...
    "summaryBlindTL", ...
    "C", ...
    "YTrueBlind", ...
    "YPredBlind", ...
    "classNames", ...
    "-v7.3");

writetable(metricsTableBlindTL, metricsCsvPath);
writetable(summaryBlindTL, summaryCsvPath);
exportgraphics(fig, confusionFigPath, "Resolution", 300);

fprintf("\nSaved transfer-learning blind-test results:\n%s\n", resultsPath);
fprintf("\nSaved blind-test metrics:\n%s\n", metricsCsvPath);
fprintf("\nSaved blind-test summary:\n%s\n", summaryCsvPath);
fprintf("\nSaved blind-test confusion matrix:\n%s\n", confusionFigPath);

fprintf("\nTransfer-learning blind-test evaluation completed successfully.\n");