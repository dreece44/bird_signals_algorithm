function results = evaluate_model(model, Xtest, Ytest)
% EVALUATE_MODEL Evaluate a trained bird species classifier.
% Inputs:
%   model - struct with clf, mu, sigma, and classNames
%   Xtest - test feature matrix (one recording per row)
%   Ytest - categorical true species labels
% Output:
%   results - struct with acc, precision, recall, and confMat

% Normalize test features using training statistics.
sigma = model.sigma;
sigma(sigma == 0) = 1;
Xtest = (Xtest - model.mu) ./ sigma;

% Predict species for the test recordings.
Ypred = predict(model.clf, Xtest);
Ypred = categorical(string(Ypred));
Ytest = categorical(string(Ytest));

% Keep the same class order in every confusion matrix.
classNames = string(model.classNames);
order = categorical(classNames);
confMat = confusionmat(Ytest, Ypred, 'Order', order);

% Calculate overall accuracy and per-species scores.
results.acc = sum(diag(confMat)) / sum(confMat(:));
correct = diag(confMat);
predictedCount = sum(confMat, 1)';
actualCount = sum(confMat, 2);
results.precision = correct ./ max(predictedCount, 1);
results.recall = correct ./ max(actualCount, 1);
results.confMat = confMat;

% Display the confusion matrix.
confusionchart(confMat, cellstr(classNames));
title('Bird Species Confusion Matrix');
end
