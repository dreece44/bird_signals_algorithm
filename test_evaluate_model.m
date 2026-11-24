
% Test evaluate_model.m with synthetic data
clear;
clc;
close all;

rng(42);

% Create sample data for three bird species
X = [randn(20, 4);
    randn(20, 4) + 3;
    randn(20, 4) + 6];

Y = categorical([repmat("Robin", 20, 1);
    repmat("Sparrow", 20, 1);
    repmat("Cardinal", 20, 1)]);

% Split into training and testing data
cv = cvpartition(Y, 'HoldOut', 0.2);

Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));

Xtest = X(test(cv), :);
Ytest = Y(test(cv));

% Normalize using training data only
mu = mean(Xtrain, 1);
sigma = std(Xtrain, 0, 1);
sigma(sigma == 0) = 1;

XtrainNorm = (Xtrain - mu) ./ sigma;

% Train a temporary classifier
clf = fitcecoc(XtrainNorm, Ytrain);

% Create the expected model structure
model.clf = clf;
model.mu = mu;
model.sigma = sigma;
model.classNames = categories(Ytrain);

% Run your evaluation function
results = evaluate_model(model, Xtest, Ytest);

% Display results
disp(results);
