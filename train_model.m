function model = train_model(Xtrain, Ytrain, method)

if size(Xtrain, 1) ~= numel(Ytrain)
    error('Features and labels must have matching rows.');
end

if any(~isfinite(Xtrain(:)))
    error('Training features must contain finite values.');
end

Ytrain = categorical(Ytrain(:));

mu = mean(Xtrain, 1);
sigma = std(Xtrain, 0, 1);
sigma(sigma == 0) = 1;

Xnorm = (Xtrain - mu) ./ sigma;

switch lower(string(method))

    case "svm"
        svmTemplate = templateSVM( ...
            'KernelFunction', 'gaussian', ...
            'KernelScale', 'auto', ...
            'Standardize', false);

        clf = fitcecoc(Xnorm, Ytrain, ...
            'Learners', svmTemplate);

    case "ensemble"
        treeTemplate = templateTree( ...
            'MaxNumSplits', 20);

        clf = fitcensemble(Xnorm, Ytrain, ...
            'Method', 'Bag', ...
            'NumLearningCycles', 100, ...
            'Learners', treeTemplate);

    otherwise
        error('Unknown method. Use svm or ensemble.');
end

cvModel = crossval(clf, 'KFold', 5);
cvLoss = kfoldLoss(cvModel);

fprintf('%s CV accuracy: %.2f%%\n', ...
    upper(string(method)), (1 - cvLoss) * 100);

model.clf = clf;
model.mu = mu;
model.sigma = sigma;
model.classNames = categories(Ytrain);
model.cvLoss = cvLoss;

end
