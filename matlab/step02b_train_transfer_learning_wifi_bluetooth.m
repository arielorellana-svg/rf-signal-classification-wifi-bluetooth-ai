clear; clc; close all;

%% ============================================================
%  STEP 02B - FINAL TRANSFER LEARNING TRAINING
%  RF Signal Classification: WiFi / Bluetooth / Coexistence / Noise / Unknown
%
%  Base network: ResNet-18 pretrained on ImageNet
%
%  This script provides the transfer-learning workflow required by the
%  project review. It keeps step02_train_cnn_wifi_bluetooth.m as the
%  custom CNN baseline and trains a second model using a pretrained network.
%
%  Author: Ariel Orellana
%% ============================================================

%% -----------------------------
%  Project configuration
%% -----------------------------
projectDir = "C:\Users\DETPC\Desktop\Proyecto";

dataDir = fullfile(projectDir, "data", "spectrograms_v3_domain_randomized");
modelsDir = fullfile(projectDir, "models");
resultsDir = fullfile(projectDir, "results");
checkpointDir = fullfile(projectDir, "results", "checkpoints_transfer_learning");

if ~exist(dataDir, "dir")
    error("Dataset folder not found: %s", dataDir);
end

if ~exist(modelsDir, "dir")
    mkdir(modelsDir);
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

if ~exist(checkpointDir, "dir")
    mkdir(checkpointDir);
end

%% -----------------------------
%  Training parameters
%% -----------------------------
rng(2026);

baseNetworkName = "resnet18";

trainRatio = 0.70;
validationRatioFromRemaining = 0.50;   % 15% validation, 15% internal test

maxEpochs = 12;
miniBatchSize = 64;

initialLearnRate = 1e-4;
l2Regularization = 1e-4;

validationFrequency = 50;
validationPatience = 6;

%% -----------------------------
%  Load spectrogram dataset
%% -----------------------------
imds = imageDatastore(dataDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Dataset count by class:");
disp(countEachLabel(imds));

classNames = categories(imds.Labels);
numClasses = numel(classNames);

fprintf("\nNumber of classes: %d\n", numClasses);
disp("Class names:");
disp(classNames);

%% -----------------------------
%  Split dataset: 70 / 15 / 15
%% -----------------------------
[imdsTrain, imdsTemp] = splitEachLabel(imds, trainRatio, "randomized");
[imdsVal, imdsTest] = splitEachLabel(imdsTemp, validationRatioFromRemaining, "randomized");

disp("Training set:");
disp(countEachLabel(imdsTrain));

disp("Validation set:");
disp(countEachLabel(imdsVal));

disp("Internal test set:");
disp(countEachLabel(imdsTest));

%% -----------------------------
%  Load pretrained ResNet-18
%% -----------------------------
netBase = resnet18;
inputSize = netBase.Layers(1).InputSize;

fprintf("\nLoaded pretrained network: %s\n", baseNetworkName);
fprintf("Input size: %d x %d x %d\n", inputSize(1), inputSize(2), inputSize(3));

%% -----------------------------
%  Replace final classification layers
%% -----------------------------
lgraph = layerGraph(netBase);

newFCLayer = fullyConnectedLayer(numClasses, ...
    "Name", "rf_fc", ...
    "WeightLearnRateFactor", 10, ...
    "BiasLearnRateFactor", 10);

newClassLayer = classificationLayer("Name", "rf_classoutput");

% Standard final layer names for MATLAB ResNet-18.
lgraph = replaceLayer(lgraph, "fc1000", newFCLayer);
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", newClassLayer);

%% -----------------------------
%  Data augmentation
%% -----------------------------
% Spectrogram axes represent time and frequency. Therefore, avoid rotation
% and flipping. Use only small translations and small scale variations.
augmenter = imageDataAugmenter( ...
    "RandXTranslation", [-4 4], ...
    "RandYTranslation", [-4 4], ...
    "RandXScale", [0.98 1.02], ...
    "RandYScale", [0.98 1.02]);

% Your images are grayscale 224 x 224. ResNet-18 requires 224 x 224 x 3.
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    "ColorPreprocessing", "gray2rgb", ...
    "DataAugmentation", augmenter);

augimdsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    "ColorPreprocessing", "gray2rgb");

augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    "ColorPreprocessing", "gray2rgb");

%% -----------------------------
%  Training options
%% -----------------------------
% ValidationPatience enables early stopping when validation performance
% stops improving. MathWorks documents that ValidationPatience controls how
% many validations can be worse than the previous best before stopping.
% OutputNetwork is set to return the best validation model when supported.

try
    options = trainingOptions("adam", ...
        "InitialLearnRate", initialLearnRate, ...
        "MaxEpochs", maxEpochs, ...
        "MiniBatchSize", miniBatchSize, ...
        "Shuffle", "every-epoch", ...
        "ValidationData", augimdsVal, ...
        "ValidationFrequency", validationFrequency, ...
        "ValidationPatience", validationPatience, ...
        "OutputNetwork", "best-validation", ...
        "L2Regularization", l2Regularization, ...
        "CheckpointPath", checkpointDir, ...
        "Verbose", true, ...
        "Plots", "training-progress", ...
        "ExecutionEnvironment", "auto");
catch
    warning("OutputNetwork='best-validation' was not accepted by this MATLAB version. Using default OutputNetwork.");
    
    options = trainingOptions("adam", ...
        "InitialLearnRate", initialLearnRate, ...
        "MaxEpochs", maxEpochs, ...
        "MiniBatchSize", miniBatchSize, ...
        "Shuffle", "every-epoch", ...
        "ValidationData", augimdsVal, ...
        "ValidationFrequency", validationFrequency, ...
        "ValidationPatience", validationPatience, ...
        "L2Regularization", l2Regularization, ...
        "CheckpointPath", checkpointDir, ...
        "Verbose", true, ...
        "Plots", "training-progress", ...
        "ExecutionEnvironment", "auto");
end

%% -----------------------------
%  Train transfer-learning model
%% -----------------------------
fprintf("\nStarting final transfer learning training...\n");

tic;
[netTL, trainInfo] = trainNetwork(augimdsTrain, lgraph, options);
trainingTimeSeconds = toc;

fprintf("\nTransfer learning training completed.\n");
fprintf("Training time: %.2f minutes\n", trainingTimeSeconds / 60);

%% -----------------------------
%  Evaluate on validation set
%% -----------------------------
fprintf("\nEvaluating transfer-learning model on validation set...\n");

YPredVal = classify(netTL, augimdsVal, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");

YTrueVal = imdsVal.Labels;
accuracyVal = mean(YPredVal == YTrueVal);

fprintf("Transfer learning validation accuracy: %.2f %%\n", accuracyVal * 100);

%% -----------------------------
%  Evaluate on internal test set
%% -----------------------------
fprintf("\nEvaluating transfer-learning model on internal test set...\n");

YPredTest = classify(netTL, augimdsTest, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");

YTrueTest = imdsTest.Labels;
accuracyTL = mean(YPredTest == YTrueTest);

fprintf("\nTransfer learning internal test accuracy: %.2f %%\n", accuracyTL * 100);

%% -----------------------------
%  Confusion matrix
%% -----------------------------
classOrder = categorical(classNames, classNames);

fig = figure;
cm = confusionchart(YTrueTest, YPredTest, ...
    "Order", classOrder);

cm.Title = "Confusion Matrix - ResNet-18 Transfer Learning";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

%% -----------------------------
%  Metrics by class
%% -----------------------------
classOrder = categorical(classNames, classNames);

C = confusionmat(YTrueTest, YPredTest, ...
    "Order", classOrder);

fig = figure;
cm = confusionchart(C, classOrder);

cm.Title = "Confusion Matrix - ResNet-18 Transfer Learning";
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

metricsTableTL = table( ...
    string(classNames(:)), ...
    precision(:), ...
    recall(:), ...
    f1score(:), ...
    'VariableNames', {'Class','Precision','Recall','F1_score'});

disp("Transfer learning metrics on internal test set:");
disp(metricsTableTL);

%% -----------------------------
%  Training summary table
%% -----------------------------
summaryTable = table( ...
    string(baseNetworkName), ...
    numClasses, ...
    maxEpochs, ...
    miniBatchSize, ...
    initialLearnRate, ...
    l2Regularization, ...
    validationPatience, ...
    accuracyVal, ...
    accuracyTL, ...
    trainingTimeSeconds, ...
    'VariableNames', { ...
        'BaseNetwork', ...
        'NumClasses', ...
        'MaxEpochs', ...
        'MiniBatchSize', ...
        'InitialLearnRate', ...
        'L2Regularization', ...
        'ValidationPatience', ...
        'ValidationAccuracy', ...
        'InternalTestAccuracy', ...
        'TrainingTimeSeconds'});

disp("Training summary:");
disp(summaryTable);

%% -----------------------------
%  Save model and results
%% -----------------------------
modelPath = fullfile(modelsDir, "resnet18_transfer_learning_wifi_bluetooth.mat");

metricsPath = fullfile(resultsDir, ...
    "metrics_transfer_learning_internal_test.csv");

summaryPath = fullfile(resultsDir, ...
    "summary_transfer_learning_training.csv");

confusionPath = fullfile(resultsDir, ...
    "confusion_matrix_transfer_learning_internal_test.png");

trainingInfoPath = fullfile(resultsDir, ...
    "training_info_transfer_learning.mat");

save(modelPath, ...
    "netTL", ...
    "accuracyTL", ...
    "accuracyVal", ...
    "metricsTableTL", ...
    "summaryTable", ...
    "classNames", ...
    "C", ...
    "inputSize", ...
    "baseNetworkName", ...
    "maxEpochs", ...
    "miniBatchSize", ...
    "initialLearnRate", ...
    "l2Regularization", ...
    "validationPatience", ...
    "trainingTimeSeconds", ...
    "-v7.3");

save(trainingInfoPath, "trainInfo", "-v7.3");

writetable(metricsTableTL, metricsPath);
writetable(summaryTable, summaryPath);
exportgraphics(fig, confusionPath, "Resolution", 300);

fprintf("\nSaved transfer-learning model:\n%s\n", modelPath);
fprintf("\nSaved metrics:\n%s\n", metricsPath);
fprintf("\nSaved training summary:\n%s\n", summaryPath);
fprintf("\nSaved training information:\n%s\n", trainingInfoPath);
fprintf("\nSaved confusion matrix:\n%s\n", confusionPath);

fprintf("\nFinal transfer learning training completed successfully.\n");
