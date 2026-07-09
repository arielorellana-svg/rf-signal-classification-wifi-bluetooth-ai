clear; clc; close all;

%% ============================================================
%  EVALUACIÓN FINAL INDEPENDIENTE
%  Modelo: CNN V3 domain-randomized
%  Dataset: blind_test_v2_final
%% ============================================================

%% Cargar modelo V3
modelPath = fullfile(pwd, "models", "cnn_wifi_bluetooth_v3_domain_randomized.mat");

if ~exist(modelPath, "file")
    error("No se encontró el modelo V3 en: %s", modelPath);
end

load(modelPath, "net");

%% Cargar dataset final
blindDir = fullfile(pwd, "data", "blind_test_v2_final");

if ~exist(blindDir, "dir")
    error("No se encontró el dataset final en: %s", blindDir);
end

imdsFinal = imageDatastore(blindDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Conteo del dataset final independiente:");
disp(countEachLabel(imdsFinal));

%% Clasificación
YPred = classify(net, imdsFinal, ...
    "MiniBatchSize", 64, ...
    "ExecutionEnvironment", "auto");

YTrue = imdsFinal.Labels;

finalAccuracy = mean(YPred == YTrue);

fprintf("\nAccuracy en prueba final independiente: %.2f %%\n", finalAccuracy*100);

%% Matriz de confusión
fig = figure;
cm = confusionchart(YTrue, YPred);
cm.Title = "Matriz de confusión - Prueba final independiente";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

%% Métricas por clase
classNames = string(categories(YTrue));
classNames = classNames(:);

C = confusionmat(YTrue, YPred);

TP = diag(C);
FP = sum(C,1)' - TP;
FN = sum(C,2) - TP;

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
f1score = 2 * (precision .* recall) ./ (precision + recall);

precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1score(isnan(f1score)) = 0;

metricsTableFinal = table( ...
    classNames, ...
    precision(:), ...
    recall(:), ...
    f1score(:));

metricsTableFinal.Properties.VariableNames = { ...
    'Clase', ...
    'Precision', ...
    'Recall', ...
    'F1_score'};

disp("Métricas por clase en prueba final independiente:");
disp(metricsTableFinal);

%% Guardar resultados
resultsDir = fullfile(pwd, "results");

if ~exist(resultsDir, "dir")
    mkdir(resultsDir);
end

resultsPath = fullfile(resultsDir, "blind_test_results_v3_final.mat");
metricsCsvPath = fullfile(resultsDir, "metrics_blind_test_v3_final.csv");
confusionFigPath = fullfile(resultsDir, "confusion_matrix_blind_test_v3_final.png");

save(resultsPath, ...
    "finalAccuracy", ...
    "metricsTableFinal", ...
    "C", ...
    "YTrue", ...
    "YPred");

writetable(metricsTableFinal, metricsCsvPath);

exportgraphics(fig, confusionFigPath, "Resolution", 300);

disp("Resultados guardados en:");
disp(resultsPath);

disp("Métricas CSV guardadas en:");
disp(metricsCsvPath);

disp("Imagen de matriz de confusión guardada en:");
disp(confusionFigPath);
