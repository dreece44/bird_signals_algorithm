
% main.m
% Bird Species Identification Using Digital Signal Processing
% Main program for loading, processing, training, and evaluation.

clear;
clc;
close all;

rng(42);

% Project directories
projectDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(projectDir, 'data');
resultsDir = fullfile(projectDir, 'results');

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Check for required project functions
requiredFunctions = {
    'load_dataset'
    'split_data'
    'preprocess_audio'
    'extract_features'
    'train_model'
    'evaluate_model'
    'test_noise_robustness'
    };

for i = 1:numel(requiredFunctions)
    if exist(requiredFunctions{i}, 'file') ~= 2
        error('Missing required function: %s.m', ...
            requiredFunctions{i});
    end
end

% Load dataset and recording groups
fprintf('Loading bird recordings...\n');
[files, labels, groups] = load_dataset(dataDir);

% Split by original recording to prevent data leakage
[trainIdx, testIdx] = split_data(labels, 0.20, 42, groups);

% Extract features for all recordings
fprintf('Extracting audio features...\n');

X = [];

for i = 1:numel(files)
    [audio, fs] = audioread(files{i});
    [audio, fs] = preprocess_audio(audio, fs);

    features = extract_features(audio, fs);
    X(i, :) = features;
end

% Separate training and testing data
Xtrain = X(trainIdx, :);
Ytrain = labels(trainIdx);

% Keep recording groups for cross-validation
trainGroups = groups(trainIdx);

Xtest = X(testIdx, :);
Ytest = labels(testIdx);

testFiles = files(testIdx);

% Train and evaluate classifiers
methods = {'svm', 'ensemble'};

for i = 1:numel(methods)
    method = methods{i};

    fprintf('Training %s classifier...\n', method);

    
    model = train_model(Xtrain, Ytrain, method, trainGroups);
    results = evaluate_model(model, Xtest, Ytest);

    fprintf('%s accuracy: %.2f%%\n', ...
        upper(method), results.acc * 100);

    % Save results and confusion matrix
    resultsToSave = rmfield(results, 'fig');
    save(fullfile(resultsDir, ...
        [method '_results.mat']), 'resultsToSave');

    exportgraphics(results.fig, fullfile(resultsDir, ...
        [method '_confusion_matrix.png']));

    noiseResults = test_noise_robustness(model, testFiles, Ytest);

    % Save noise robustness results
    noiseResultsToSave = rmfield(noiseResults, 'fig');
    save(fullfile(resultsDir, ...
        [method '_noise_results.mat']), 'noiseResultsToSave');

    % Save accuracy vs SNR figure
    exportgraphics(noiseResults.fig, fullfile(resultsDir, ...
        [method '_noise_robustness.png']));
end

fprintf('Evaluation complete.\n');
