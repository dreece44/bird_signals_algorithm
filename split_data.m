function [trainIdx, testIdx] = split_data(labels, testRatio, seed, groups)
% SPLIT_DATA Stratified train/test hold-out split, by recording.
% With groups, whole recordings go to one side: per species, about
% testRatio of its recordings go to test, and every clip of a test
% recording follows it. Clips from one recording share the bird,
% microphone and background, so splitting by clip inflates accuracy.
% Without groups it falls back to a clip-level stratified cvpartition.
%
% Inputs:
%   labels    - N-by-1 categorical species labels
%   testRatio - Fraction held out for testing (default 0.2)
%   seed      - Random seed for reproducibility (default 42)
%   groups    - Optional N-by-1 split groups from load_dataset
%
% Outputs:
%   trainIdx - N-by-1 logical, true for training clips
%   testIdx  - N-by-1 logical, true for test clips (~trainIdx)

if nargin < 2 || isempty(testRatio), testRatio = 0.2; end
if nargin < 3 || isempty(seed),      seed = 42;       end
labels = categorical(labels(:));
rng(seed);

if nargin < 4 || isempty(groups)
    warning('split_data:noGroups', ['No groups given: splitting by ' ...
        'clip, so one recording can land in both sets.']);
    cv = cvpartition(labels, 'HoldOut', testRatio);
    trainIdx = training(cv);
    testIdx  = test(cv);
    return
end

groups = string(groups(:));
if numel(groups) ~= numel(labels)
    error('split_data:size', 'labels and groups must have the same length.');
end

% Pick test recordings species by species
testGroups = strings(0, 1);
for s = categories(labels)'
    g = unique(groups(labels == s{1}));
    if numel(g) < 2
        warning('split_data:oneGroup', ['%s has only one recording; ' ...
            'all of it goes to training.'], s{1});
        continue
    end
    g = g(randperm(numel(g)));
    nTest = min(max(1, round(testRatio * numel(g))), numel(g) - 1);
    testGroups = [testGroups; g(1:nTest)]; %#ok<AGROW>
end

% Every clip of a test recording goes to test
testIdx  = ismember(groups, testGroups);
trainIdx = ~testIdx;

fprintf('Split: %d train / %d test clips (%d test recordings).\n', ...
    nnz(trainIdx), nnz(testIdx), numel(testGroups));
end
