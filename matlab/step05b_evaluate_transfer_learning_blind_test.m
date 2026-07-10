clear; clc; close all;

%% ============================================================
% STEP 05B - EVALUATE TRANSFER LEARNING MODEL ON FINAL BLIND TEST
% Model: ResNet-18 transfer learning
% Dataset: blind_test_v2_final
%% ============================================================

%% Project paths
scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

modelPath = fullfile(projectDir, "models", ...
    "resnet18_transfer_learning_wifi_bluetooth.mat");

blindCandidates = [ ...
    fullfile(projectDir, "data", "blind_test_v2_final"), ...
    fullfile(projectDir, "results", "blindtestv2_final"), ...
    fullfile(projectDir, "results", "blind_test_v2_final")];

blindDir = firstExistingFolder(blindCandidates);
resultsDir = fullfile(projectDir, "results");

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end

if strlength(blindDir) == 0
    error(["Blind-test dataset not found. Expected one of:\n%s"], ...
        strjoin(blindCandidates, newline));
end

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

fprintf("Using blind-test dataset:\n%s\n", blindDir);

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

%% Load final blind-test dataset
imdsBlind = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Blind-test dataset count by class:");
disp(countEachLabel(imdsBlind));

%% Check class consistency
blindClassNames = categories(imdsBlind.Labels);

if ~isequal(string(classNames(:)), string(blindClassNames(:)))
    warning("Class order or names differ. Metrics will use the model class order.");
end

%% Prepare grayscale spectrograms for ResNet-18 RGB input
augimdsBlind = augmentedImageDatastore(inputSize(1:2), imdsBlind, ...
    "ColorPreprocessing", "gray2rgb");

%% Classification
miniBatchSize = 64;
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
cm.Title = "Confusion Matrix - ResNet-18 Final Blind Test";
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
    string(classNames(:)), precision(:), recall(:), f1score(:), ...
    'VariableNames', {'Class','Precision','Recall','F1_score'});

summaryBlindTL = table( ...
    string("resnet18_transfer_learning"), ...
    numel(classNames), ...
    numel(imdsBlind.Files), ...
    finalBlindAccuracyTL, ...
    miniBatchSize, ...
    string(relativePath(projectDir, blindDir)), ...
    'VariableNames', { ...
        'Model','NumClasses','NumBlindTestSamples', ...
        'FinalBlindTestAccuracy','MiniBatchSize','BlindTestDatasetPath'});

disp(metricsTableBlindTL);
disp(summaryBlindTL);

%% Save results
resultsPath = fullfile(resultsDir, "blind_test_results_transfer_learning.mat");
metricsCsvPath = fullfile(resultsDir, "metrics_transfer_learning_blind_test.csv");
summaryCsvPath = fullfile(resultsDir, "summary_transfer_learning_blind_test.csv");
confusionFigPath = fullfile(resultsDir, "confusion_matrix_transfer_learning_blind_test.png");

save(resultsPath, "finalBlindAccuracyTL", "metricsTableBlindTL", ...
    "summaryBlindTL", "C", "YTrueBlind", "YPredBlind", ...
    "classNames", "-v7.3");

writetable(metricsTableBlindTL, metricsCsvPath);
writetable(summaryBlindTL, summaryCsvPath);
exportgraphics(fig, confusionFigPath, "Resolution", 300);

fprintf("\nTransfer-learning blind-test evaluation completed successfully.\n");

%% Local helpers
function folder = firstExistingFolder(candidates)
    folder = "";
    for k = 1:numel(candidates)
        if exist(candidates(k), "dir") == 7
            folder = string(candidates(k));
            return;
        end
    end
end

function p = relativePath(projectDir, absolutePath)
    prefix = string(projectDir) + filesep;
    p = erase(string(absolutePath), prefix);
end