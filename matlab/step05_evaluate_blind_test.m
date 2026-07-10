clc; close all;

%% ============================================================
% STEP 05 - EVALUATE CUSTOM CNN ON THE INDEPENDENT BLIND TEST
%% ============================================================

scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

modelPath = fullfile(projectDir, "models", ...
    "cnn_wifi_bluetooth_v3_domain_randomized.mat");
resultsDir = fullfile(projectDir, "results");

expectedClasses = ["Bluetooth", "Noise", "Unknown", "WiFi", ...
    "WiFi_Bluetooth_Overlap", "WiFi_Bluetooth_Separated"];

blindCandidates = [ ...
    fullfile(projectDir, "data", "blind_test_v2_final"), ...
    fullfile(projectDir, "results", "blindtestv2_final"), ...
    fullfile(projectDir, "results", "blind_test_v2_final")];

blindDir = findClassFolder(blindCandidates, expectedClasses);

if ~exist(modelPath, "file")
    error("Custom CNN model not found: %s", modelPath);
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
if ~isfield(modelData, "net")
    error("The model file does not contain variable 'net': %s", modelPath);
end
net = modelData.net;

if isfield(modelData, "classNames")
    classNames = string(modelData.classNames(:));
else
    classNames = expectedClasses(:);
end

imdsBlind = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

validateClassSet(imdsBlind.Labels, expectedClasses);
disp("Independent blind-test count by class:");
disp(countEachLabel(imdsBlind));

miniBatchSize = 64;
YPredBlind = classify(net, imdsBlind, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");
YTrueBlind = imdsBlind.Labels;
blindTestAccuracyCNN = mean(YPredBlind == YTrueBlind);

fprintf("\nCustom CNN independent blind-test accuracy: %.2f %%\n", ...
    blindTestAccuracyCNN * 100);

classOrder = categorical(classNames, classNames);
confusionMatrix = confusionmat(YTrueBlind, YPredBlind, "Order", classOrder);

fig = figure;
chart = confusionchart(confusionMatrix, categorical(classNames, classNames));
chart.Title = "Confusion Matrix - Custom CNN Independent Blind Test";
chart.RowSummary = "row-normalized";
chart.ColumnSummary = "column-normalized";

[precision, recall, f1Score] = classificationMetrics(confusionMatrix);
metricsTableBlindCNN = table(classNames, precision, recall, f1Score, ...
    "VariableNames", {"Class","Precision","Recall","F1_score"});

summaryBlindCNN = table( ...
    "custom_cnn_baseline", numel(classNames), numel(imdsBlind.Files), ...
    blindTestAccuracyCNN, miniBatchSize, relativePath(projectDir, blindDir), ...
    "VariableNames", {"Model","NumClasses","NumBlindTestSamples", ...
    "BlindTestAccuracy","MiniBatchSize","BlindTestDatasetPath"});

disp(metricsTableBlindCNN);
disp(summaryBlindCNN);

save(fullfile(resultsDir, "blind_test_results_cnn_baseline.mat"), ...
    "blindTestAccuracyCNN", "metricsTableBlindCNN", "summaryBlindCNN", ...
    "confusionMatrix", "YTrueBlind", "YPredBlind", "classNames", ...
    "blindDir", "-v7.3");
writetable(metricsTableBlindCNN, ...
    fullfile(resultsDir, "metrics_cnn_blind_test.csv"));
writetable(summaryBlindCNN, ...
    fullfile(resultsDir, "summary_cnn_blind_test.csv"));
exportgraphics(fig, ...
    fullfile(resultsDir, "confusion_matrix_cnn_blind_test.png"), ...
    "Resolution", 300);

fprintf("\nCustom-CNN blind-test evaluation completed successfully.\n");

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
