clearvars; clc; close all;

%% ============================================================
% STEP 07 - RUN ALL PRETRAINED MODEL EVALUATIONS
%% ============================================================

scriptPath = mfilename("fullpath");
matlabDir = fileparts(scriptPath);

scripts = [ ...
    "step05_evaluate_blind_test.m", ...
    "step05b_evaluate_transfer_learning_blind_test.m", ...
    "step06_evaluate_cnn_receiver_like.m", ...
    "step06b_evaluate_transfer_learning_receiver_like.m"];

fprintf("Running %d evaluation scripts...\n\n", numel(scripts));

for k = 1:numel(scripts)
    scriptFile = fullfile(matlabDir, scripts(k));
    fprintf("[%d/%d] %s\n", k, numel(scripts), scripts(k));
    run(scriptFile);
    fprintf("Completed: %s\n\n", scripts(k));
end

fprintf("All pretrained-model evaluations completed successfully.\n");
