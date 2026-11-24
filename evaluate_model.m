
function results = evaluate_model(model, Xtest, Ytest, showPlot)
% EVALUATE_MODEL Evaluate a trained bird species classifier.
% Inputs:
%   model    - Struct with clf, mu, sigma, and classNames
%   Xtest    - Test feature matrix (one recording per row)
%   Ytest    - True species labels
%   showPlot - Optional flag to display confusion matrix
%
% Output:
%   results - Struct with acc, precision, recall, confMat, and fig

if nargin < 4
    showPlot = true;
end

if isempty(Xtest) || isempty(Ytest)
    error('Test data cannot be empty.');
end

if size(Xtest, 1) ~= numel(Ytest)
    error('Number of test recordings and labels must match.');
end

% Normalize using training statistics
sigma = model.sigma;
sigma(sigma == 0) = 1;
Xtest = (Xtest - model.mu) ./ sigma;

% Predict species
Ypred = predict(model.clf, Xtest);
Ypred = categorical(string(Ypred));
Ytest = categorical(string(Ytest));

% Keep class order consistent
classNames = string(model.classNames);
order = categorical(classNames);

confMat = confusionmat(Ytest, Ypred, 'Order', order);

% Calculate evaluation metrics
correct = diag(confMat);
predictedCount = sum(confMat, 1)';
actualCount = sum(confMat, 2);

results.acc = sum(correct) / sum(confMat(:));
results.precision = correct ./ max(predictedCount, 1);
results.recall = correct ./ max(actualCount, 1);
results.confMat = confMat;
results.fig = [];

% Display confusion matrix if requested
if showPlot
    results.fig = figure;
    confusionchart(confMat, cellstr(classNames));
    title('Bird Species Confusion Matrix');
end
end
