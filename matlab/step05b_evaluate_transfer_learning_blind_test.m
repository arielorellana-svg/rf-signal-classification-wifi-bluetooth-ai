clc; close all;

%% ============================================================
% STEP 05B - EVALUATE RESNET-18 ON THE INDEPENDENT BLIND TEST
%% ============================================================

scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

modelPath = fullfile(projectDir, "models", ...
    "resnet18_transfer_learning_wifi_bluetooth.mat");
resultsDir = fullfile(projectDir, "results");

expectedClasses = ["Bluetooth", "Noise", "Unknown", "WiFi", ...
    "WiFi_Bluetooth_Overlap", "WiFi_Bluetooth_Separated"];

blindCandidates = [ ...
    fullfile(projectDir, "data", "blind_test_v2_final"), ...
    fullfile(projectDir, "results", "blindtestv2_final"), ...
    fullfile(projectDir, "results", "blind_test_v2_final")];

blindDir = findClassFolder(blindCandidates, expectedClasses);

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end
if strlength(blindDir) == 0
    error("Independent blind-test dataset not found. Generate it with step03_generate_blind_test.m or check:\n%s", ...
        strjoin(blindCandidates, newline));
end
if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

fprintf("Using independent blind-test dataset:\n%s\n", blindDir);

modelData = load(modelPath);
if ~isfield(modelData, "netTL")
    error("The model file does not contain variable 'netTL': %s", modelPath);
end
netTL = modelData.netTL;

if isfield(modelData, "classNames")
    classNames = string(modelData.classNames(:));
else
    classNames = expectedClasses(:);
end
if isfield(modelData, "inputSize")
    inputSize = modelData.inputSize;
else
    inputSize = [224 224 3];
end

imdsBlind = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

validateClassSet(imdsBlind.Labels, expectedClasses);
disp("Independent blind-test count by class:");
disp(countEachLabel(imdsBlind));

augimdsBlind = augmentedImageDatastore(inputSize(1:2), imdsBlind, ...
    "ColorPreprocessing", "gray2rgb");

miniBatchSize = 64;
YPredBlind = classify(netTL, augimdsBlind, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");
YTrueBlind = imdsBlind.Labels;
blindTestAccuracyTL = mean(YPredBlind == YTrueBlind);

fprintf("\nResNet-18 independent blind-test accuracy: %.2f %%\n", ...
    blindTestAccuracyTL * 100);

classOrder = categorical(classNames, classNames);
confusionMatrix = confusionmat(YTrueBlind, YPredBlind, "Order", classOrder);

fig = figure;
chart = confusionchart(confusionMatrix, categorical(classNames, classNames));
chart.Title = "Confusion Matrix - ResNet-18 Independent Blind Test";
chart.RowSummary = "row-normalized";
chart.ColumnSummary = "column-normalized";

[precision, recall, f1Score] = classificationMetrics(confusionMatrix);
metricsTableBlindTL = table(classNames, precision, recall, f1Score, ...
    "VariableNames", {"Class","Precision","Recall","F1_score"});

summaryBlindTL = table( ...
    "resnet18_transfer_learning", numel(classNames), ...
    numel(imdsBlind.Files), blindTestAccuracyTL, miniBatchSize, ...
    relativePath(projectDir, blindDir), ...
    "VariableNames", {"Model","NumClasses","NumBlindTestSamples", ...
    "BlindTestAccuracy","MiniBatchSize","BlindTestDatasetPath"});

disp(metricsTableBlindTL);
disp(summaryBlindTL);

save(fullfile(resultsDir, "blind_test_results_transfer_learning.mat"), ...
    "blindTestAccuracyTL", "metricsTableBlindTL", "summaryBlindTL", ...
    "confusionMatrix", "YTrueBlind", "YPredBlind", "classNames", ...
    "blindDir", "-v7.3");
writetable(metricsTableBlindTL, ...
    fullfile(resultsDir, "metrics_transfer_learning_blind_test.csv"));
writetable(summaryBlindTL, ...
    fullfile(resultsDir, "summary_transfer_learning_blind_test.csv"));
exportgraphics(fig, ...
    fullfile(resultsDir, "confusion_matrix_transfer_learning_blind_test.png"), ...
    "Resolution", 300);

fprintf("\nResNet-18 blind-test evaluation completed successfully.\n");

function folder = findClassFolder(candidates, expectedClasses)
    folder = "";
    for k = 1:numel(candidates)
        candidate = string(candidates(k));
        if exist(candidate, "dir") ~= 7
            continue;
        end
        hasClasses = all(arrayfun(@(className) ...
            exist(fullfile(candidate, className), "dir") == 7, expectedClasses));
        if hasClasses
            folder = candidate;
            return;
        end
    end
end

function validateClassSet(labels, expectedClasses)
    datasetClasses = string(categories(labels));
    if ~isequal(sort(datasetClasses), sort(expectedClasses(:)))
        error("Dataset classes do not match the expected six-class definition.");
    end
end

function [precision, recall, f1Score] = classificationMetrics(confusionMatrix)
    truePositive = diag(confusionMatrix);
    falsePositive = sum(confusionMatrix,1)' - truePositive;
    falseNegative = sum(confusionMatrix,2) - truePositive;

    precision = truePositive ./ (truePositive + falsePositive);
    recall = truePositive ./ (truePositive + falseNegative);
    f1Score = 2 * (precision .* recall) ./ (precision + recall);

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1Score(isnan(f1Score)) = 0;
end

function pathOut = relativePath(projectDir, absolutePath)
    pathOut = erase(string(absolutePath), string(projectDir) + filesep);
    pathOut = replace(pathOut, filesep, "/");
end
