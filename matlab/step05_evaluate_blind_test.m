clear; clc; close all;

%% ============================================================
% STEP 05 - EVALUATE CUSTOM CNN ON THE INDEPENDENT BLIND TEST
% Model: Custom CNN V3 domain-randomized
% Dataset: blind_test_v2_final
%% ============================================================

scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);
projectDir = fileparts(matlabDir);

modelPath = fullfile(projectDir, "models", ...
    "cnn_wifi_bluetooth_v3_domain_randomized.mat");
blindCandidates = [ ...
    fullfile(projectDir, "data", "blind_test_v2_final"), ...
    fullfile(projectDir, "results", "blindtestv2_final"), ...
    fullfile(projectDir, "results", "blind_test_v2_final")];
blindDir = firstExistingFolder(blindCandidates);
resultsDir = fullfile(projectDir, "results");

if ~exist(modelPath, "file")
    error("Custom CNN model not found: %s", modelPath);
end
if strlength(blindDir) == 0
    error("Blind-test dataset not found. Checked:\n%s", ...
        strjoin(blindCandidates, newline));
end
if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

fprintf("Using blind-test dataset:\n%s\n", blindDir);

S = load(modelPath);
if ~isfield(S, "net")
    error("The model file does not contain variable 'net': %s", modelPath);
end
net = S.net;

imdsFinal = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Independent blind-test count by class:");
disp(countEachLabel(imdsFinal));

miniBatchSize = 64;
YPred = classify(net, imdsFinal, ...
    "MiniBatchSize", miniBatchSize, ...
    "ExecutionEnvironment", "auto");
YTrue = imdsFinal.Labels;
finalAccuracy = mean(YPred == YTrue);

fprintf("\nCustom CNN independent blind-test accuracy: %.2f %%\n", ...
    finalAccuracy * 100);

classNames = string(categories(YTrue));
classOrder = categorical(classNames, classNames);
C = confusionmat(YTrue, YPred, 'Order', classOrder);

fig = figure;
cm = confusionchart(C, categorical(classNames, classNames));
cm.Title = "Confusion Matrix - Custom CNN Independent Blind Test";
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

metricsTableFinal = table(classNames(:), precision(:), recall(:), ...
    f1score(:), 'VariableNames', {'Class','Precision','Recall','F1_score'});

summaryBlindCNN = table( ...
    string("custom_cnn_baseline"), numel(classNames), ...
    numel(imdsFinal.Files), finalAccuracy, miniBatchSize, ...
    string(relativePath(projectDir, blindDir)), ...
    'VariableNames', {'Model','NumClasses','NumBlindTestSamples', ...
    'FinalBlindTestAccuracy','MiniBatchSize','BlindTestDatasetPath'});

disp(metricsTableFinal);
disp(summaryBlindCNN);

save(fullfile(resultsDir, "blind_test_results_v3_final.mat"), ...
    "finalAccuracy", "metricsTableFinal", "summaryBlindCNN", ...
    "C", "YTrue", "YPred");
writetable(metricsTableFinal, ...
    fullfile(resultsDir, "metrics_blind_test_v3_final.csv"));
writetable(summaryBlindCNN, ...
    fullfile(resultsDir, "summary_cnn_blind_test.csv"));
exportgraphics(fig, ...
    fullfile(resultsDir, "confusion_matrix_blind_test_v3_final.png"), ...
    "Resolution", 300);

fprintf("\nCustom CNN blind-test evaluation completed successfully.\n");

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
    p = erase(string(absolutePath), string(projectDir) + filesep);
end
