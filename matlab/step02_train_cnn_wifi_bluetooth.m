clear; clc; close all;

%% Dataset V3 domain-randomized
dataDir = fullfile(pwd, "data", "spectrograms_v3_domain_randomized");

imds = imageDatastore(dataDir, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Cantidad de imágenes por clase:");
disp(countEachLabel(imds));

%% División 70/15/15
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.70, "randomized");
[imdsVal, imdsTest] = splitEachLabel(imdsTemp, 0.50, "randomized");

disp("Entrenamiento:");
disp(countEachLabel(imdsTrain));

disp("Validación:");
disp(countEachLabel(imdsVal));

disp("Prueba:");
disp(countEachLabel(imdsTest));

%% Configuración de entrada
inputSize = [224 224 1];
numClasses = numel(categories(imds.Labels));

%% CNN V3
layers = [
    imageInputLayer(inputSize, "Name", "input")

    convolution2dLayer(3, 16, "Padding", "same", "Name", "conv_1")
    batchNormalizationLayer("Name", "bn_1")
    reluLayer("Name", "relu_1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_1")

    convolution2dLayer(3, 32, "Padding", "same", "Name", "conv_2")
    batchNormalizationLayer("Name", "bn_2")
    reluLayer("Name", "relu_2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_2")

    convolution2dLayer(3, 64, "Padding", "same", "Name", "conv_3")
    batchNormalizationLayer("Name", "bn_3")
    reluLayer("Name", "relu_3")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_3")

    convolution2dLayer(3, 128, "Padding", "same", "Name", "conv_4")
    batchNormalizationLayer("Name", "bn_4")
    reluLayer("Name", "relu_4")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_4")

    convolution2dLayer(3, 256, "Padding", "same", "Name", "conv_5")
    batchNormalizationLayer("Name", "bn_5")
    reluLayer("Name", "relu_5")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool_5")

    convolution2dLayer(3, 384, "Padding", "same", "Name", "conv_6")
    batchNormalizationLayer("Name", "bn_6")
    reluLayer("Name", "relu_6")

    globalAveragePooling2dLayer("Name", "gap")

    dropoutLayer(0.40, "Name", "dropout")

    fullyConnectedLayer(numClasses, "Name", "fc")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classification")
];

%% Opciones de entrenamiento
options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 25, ...
    "MiniBatchSize", 64, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", imdsVal, ...
    "ValidationFrequency", 50, ...
    "Verbose", true, ...
    "Plots", "training-progress", ...
    "ExecutionEnvironment", "auto");

%% Entrenar
net = trainNetwork(imdsTrain, layers, options);

%% Evaluar con prueba interna V3
YPred = classify(net, imdsTest, ...
    "MiniBatchSize", 64, ...
    "ExecutionEnvironment", "auto");

YTrue = imdsTest.Labels;

accuracy = mean(YPred == YTrue);
fprintf("\nAccuracy en prueba interna V3: %.2f %%\n", accuracy*100);

%% Matriz de confusión
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = "Matriz de confusión - Dataset V3";
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

metricsTable = table( ...
    classNames, ...
    precision(:), ...
    recall(:), ...
    f1score(:));

metricsTable.Properties.VariableNames = { ...
    'Clase', ...
    'Precision', ...
    'Recall', ...
    'F1_score'};

disp("Métricas por clase V3:");
disp(metricsTable);

%% Guardar modelo
modelsDir = fullfile(pwd, "models");

if ~exist(modelsDir, "dir")
    mkdir(modelsDir);
end

modelPath = fullfile(modelsDir, "cnn_wifi_bluetooth_v3_domain_randomized.mat");

save(modelPath, ...
    "net", ...
    "accuracy", ...
    "metricsTable", ...
    "classNames", ...
    "C");

disp("Modelo guardado en:");
disp(modelPath);
