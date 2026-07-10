clear; clc; close all;

%% ============================================================
% STEP 06 - EVALUATE RESNET-18 ON RECEIVER-LIKE VALIDATION DATA
% Dataset: independent synthetic data with receiver-inspired impairments
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
    fullfile(projectDir, "results", "data_sdr", "spectrograms"), ...
    fullfile(projectDir, "results", "data_sdr"), ...
    fullfile(projectDir, "data_simulated_sdr_v1", "spectrograms"), ...
    fullfile(projectDir, "data_simulated_sdr_v1"), ...
    fullfile(projectDir, "data_sdr", "spectrograms"), ...
    fullfile(projectDir, "data_sdr")];

validationDir = findClassFolder(validationCandidates, expectedClasses);

if ~exist(modelPath, "file")
    error("Transfer-learning model not found: %s", modelPath);
end
if strlength(validationDir) == 0
    error("Receiver-like validation dataset not found. Checked:\n%s", ...
        strjoin(validationCandidates, newline));
end
if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

fprintf("Using receiver-like validation dataset:\n%s\n", validationDir);

S = load(modelPath);
if ~isfield(S, "netTL")
    error("The model file does not contain variable 'netTL': %s", modelPath);
end
netTL = S.netTL;

if isfield(S, "classNames")
    classNames = string(S.classNames(:));
else
    classNames = expectedClasses(:);
end
if isfield(S, "inputSize")
    inputSize = S.inputSize;
else
    inputSize = [224 224 3];
end

imdsValidation = imageDatastore(validationDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Receiver-like validation count by class:");
disp(countEachLabel(imdsValidation));

validationClassNames = string(categories(imdsValidation.Labels));
if ~isequal(classNames, validationClassNames)
    warning("Class names or order differ. Metrics will use model class order.");
end

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
C = confusionmat(YTrueValidation, YPredValidation, 'Order', classOrder);

fig = figure;
cm = confusionchart(C, categorical(classNames, classNames));
cm.Title = "Confusion Matrix - ResNet-18 Receiver-Like Validation";
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

metricsTableReceiverLikeTL = table(classNames, precision(:), recall(:), ...
    f1score(:), 'VariableNames', {'Class','Precision','Recall','F1_score'});

summaryReceiverLikeTL = table( ...
    string("resnet18_transfer_learning"), numel(classNames), ...
    numel(imdsValidation.Files), receiverLikeAccuracyTL, miniBatchSize, ...
    string(relativePath(projectDir, validationDir)), ...
    'VariableNames', {'Model','NumClasses','NumReceiverLikeSamples', ...
    'ReceiverLikeAccuracy','MiniBatchSize','ReceiverLikeDatasetPath'});

disp(metricsTableReceiverLikeTL);
disp(summaryReceiverLikeTL);

save(fullfile(resultsDir, "receiver_like_results_transfer_learning.mat"), ...
    "receiverLikeAccuracyTL", "metricsTableReceiverLikeTL", ...
    "summaryReceiverLikeTL", "C", "YTrueValidation", ...
    "YPredValidation", "classNames", "validationDir", "-v7.3");
writetable(metricsTableReceiverLikeTL, ...
    fullfile(resultsDir, "metrics_transfer_learning_receiver_like.csv"));
writetable(summaryReceiverLikeTL, ...
    fullfile(resultsDir, "summary_transfer_learning_receiver_like.csv"));
exportgraphics(fig, ...
    fullfile(resultsDir, "confusion_matrix_transfer_learning_receiver_like.png"), ...
    "Resolution", 300);

fprintf("\nReceiver-like transfer-learning evaluation completed successfully.\n");

function folder = findClassFolder(candidates, expectedClasses)
    folder = "";
    for k = 1:numel(candidates)
        candidate = string(candidates(k));
        if exist(candidate, "dir") ~= 7
            continue;
        end
        hasClasses = all(arrayfun(@(c) ...
            exist(fullfile(candidate, c), "dir") == 7, expectedClasses));
        if hasClasses
            folder = candidate;
            return;
        end
    end
end

function p = relativePath(projectDir, absolutePath)
    p = erase(string(absolutePath), string(projectDir) + filesep);
end