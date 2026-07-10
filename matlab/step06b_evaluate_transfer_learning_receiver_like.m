clc; close all;

%% ============================================================
% STEP 06B - EVALUATE RESNET-18 ON RECEIVER-LIKE VALIDATION DATA
%% ============================================================

scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

modelPath = fullfile(projectDir, "models", ...
    "resnet18_transfer_learning_wifi_bluetooth.mat");
resultsDir = fullfile(projectDir, "results");

expectedClasses = ["Bluetooth", "Noise", "Unknown", "WiFi", ...
    "WiFi_Bluetooth_Overlap", "WiFi_Bluetooth_Separated"];

validationCandidates = [ ...
    fullfile(projectDir, "data", "receiver_like_validation", "spectrograms"), ...
    fullfile(projectDir, "results", "data_sdr", "spectrograms"), ...
    fullfile(projectDir, "results", "data_sdr"), ...
    fullfile(projectDir, "data_simulated_sdr_v1", "spectrograms"), ...
    fullfile(projectDir, "data_simulated_sdr_v1")];

validationDir = findClassFolder(validationCandidates, expectedClasses);

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end
if strlength(validationDir) == 0
    error("Receiver-like validation dataset not found. Generate it with step04_generate_receiver_like_validation.m or check:\n%s", ...
        strjoin(validationCandidates, newline));
end
if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

fprintf("Using receiver-like validation dataset:\n%s\n", validationDir);

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

imdsValidation = imageDatastore(validationDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

validateClassSet(imdsValidation.Labels, expectedClasses);
disp("Receiver-like validation count by class:");
disp(countEachLabel(imdsValidation));

augimdsValidation = augmentedImageDatastore(inputSize(1:2), ...
    imdsValidation, "ColorPreprocessing", "gray2rgb");

miniBatchSize = 64;
YPredValidation = classify(netTL, augimdsValidation, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");
YTrueValidation = imdsValidation.Labels;
receiverLikeAccuracyTL = mean(YPredValidation == YTrueValidation);

fprintf("\nResNet-18 receiver-like validation accuracy: %.2f %%\n", ...
    receiverLikeAccuracyTL * 100);

classOrder = categorical(classNames, classNames);
confusionMatrix = confusionmat(YTrueValidation, YPredValidation, ...
    "Order", classOrder);

fig = figure;
chart = confusionchart(confusionMatrix, categorical(classNames, classNames));
chart.Title = "Confusion Matrix - ResNet-18 Receiver-Like Validation";
chart.RowSummary = "row-normalized";
chart.ColumnSummary = "column-normalized";

[precision, recall, f1Score] = classificationMetrics(confusionMatrix);
metricsTableReceiverLikeTL = table(classNames, precision, recall, f1Score, ...
    "VariableNames", {"Class","Precision","Recall","F1_score"});

summaryReceiverLikeTL = table( ...
    "resnet18_transfer_learning", numel(classNames), ...
    numel(imdsValidation.Files), receiverLikeAccuracyTL, miniBatchSize, ...
    relativePath(projectDir, validationDir), ...
    "VariableNames", {"Model","NumClasses","NumReceiverLikeSamples", ...
    "ReceiverLikeAccuracy","MiniBatchSize","ReceiverLikeDatasetPath"});

disp(metricsTableReceiverLikeTL);
disp(summaryReceiverLikeTL);

save(fullfile(resultsDir, "receiver_like_results_transfer_learning.mat"), ...
    "receiverLikeAccuracyTL", "metricsTableReceiverLikeTL", ...
    "summaryReceiverLikeTL", "confusionMatrix", "YTrueValidation", ...
    "YPredValidation", "classNames", "validationDir", "-v7.3");
writetable(metricsTableReceiverLikeTL, ...
    fullfile(resultsDir, "metrics_transfer_learning_receiver_like.csv"));
writetable(summaryReceiverLikeTL, ...
    fullfile(resultsDir, "summary_transfer_learning_receiver_like.csv"));
exportgraphics(fig, ...
    fullfile(resultsDir, "confusion_matrix_transfer_learning_receiver_like.png"), ...
    "Resolution", 300);

fprintf("\nResNet-18 receiver-like evaluation completed successfully.\n");

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
