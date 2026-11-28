function model = train_model(Xtrain, Ytrain, method, trainGroups)


% Recording groups are optional for backward compatibility
if nargin < 4
    trainGroups = [];
end

% Verify recording group count
if ~isempty(trainGroups)
    if numel(trainGroups) ~= numel(Ytrain)
        error('Recording groups must match training labels.');
    end
end

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

% Fix random seed for reproducible model training
rng(42);

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


% Five-fold cross-validation

if isempty(trainGroups)

    % Fallback for synthetic tests without recording groups
    warning('Using clip-level CV, not recording-level CV.');

    cvModel = crossval(clf, 'KFold', 5);
    cvLoss = kfoldLoss(cvModel);

else

    % Identify unique original recordings
    trainGroups = string(trainGroups(:));

    [uniqueGroups, ~, groupIndex] = unique(trainGroups);

    % Assign a species label to each recording group
    groupSpecies = strings(numel(uniqueGroups), 1);

    for g = 1:numel(uniqueGroups)

        groupLabels = unique( ...
            string(Ytrain(groupIndex == g)));

        if numel(groupLabels) ~= 1
            error('A recording group contains multiple species.');
        end

        groupSpecies(g) = groupLabels;

    end

    % Split entire recordings into five folds
    rng(42);
    cv = cvpartition(categorical(groupSpecies), ...
        'KFold', 5);

    predictions = strings(numel(Ytrain), 1);

    for fold = 1:5

        % Select validation recordings
        valGroups = find(test(cv, fold));

        valIdx = ismember(groupIndex, valGroups);
        fitIdx = ~valIdx;

        % Normalize using only this fold's training data
        foldMu = mean(Xtrain(fitIdx, :), 1);
        foldSigma = std(Xtrain(fitIdx, :), 0, 1);
        foldSigma(foldSigma == 0) = 1;

        Xfit = (Xtrain(fitIdx, :) - foldMu) ./ foldSigma;
        Xval = (Xtrain(valIdx, :) - foldMu) ./ foldSigma;

        % Train the selected classifier
        switch lower(string(method))

            case "svm"
                learner = templateSVM( ...
                    'KernelFunction', 'gaussian', ...
                    'KernelScale', 'auto', ...
                    'Standardize', false);

                foldClf = fitcecoc( ...
                    Xfit, Ytrain(fitIdx), ...
                    'Learners', learner);

            case "ensemble"
                learner = templateTree( ...
                    'MaxNumSplits', 20);

                foldClf = fitcensemble( ...
                    Xfit, Ytrain(fitIdx), ...
                    'Method', 'Bag', ...
                    'NumLearningCycles', 100, ...
                    'Learners', learner);

        end

        % Predict held-out recording clips
        predictions(valIdx) = string(predict(foldClf, Xval));

    end

    % Calculate accuracy across all validation clips
    cvLoss = mean(predictions ~= string(Ytrain));

end

fprintf('%s CV accuracy: %.2f%%\n', ...
    upper(string(method)), (1 - cvLoss) * 100);

model.clf = clf;
model.mu = mu;
model.sigma = sigma;
model.classNames = categories(Ytrain);
model.cvLoss = cvLoss;

% SVM scores are not probabilities. A second, posterior-fitted copy is
% used only for predict_species' confidence score; model.clf (which makes
% every prediction) is left unchanged, since FitPosterior also changes
% how fitcecoc picks the class.
model.scoreClf = [];
if lower(string(method)) == "svm"
    model.scoreClf = fitcecoc(Xnorm, Ytrain, ...
        'Learners', svmTemplate, 'FitPosterior', true);
end

end
