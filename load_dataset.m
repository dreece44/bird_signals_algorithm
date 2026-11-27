function [files, labels, groups, info] = load_dataset(dataDir, vocalization)
% LOAD_DATASET Load bird clip paths and species labels from manifest.csv.
% Labels come from the manifest's species column, not folder names:
% clips sit in data/<Species>/<song|call>/Recording<N>/, so
% audioDatastore's 'foldernames' would label them "Recording1", ...
%
% Inputs:
%   dataDir      - Folder holding manifest.csv (default: data/ next to
%                  this file)
%   vocalization - "song" (default), "call", or "all". Use songs for the
%                  species classifier (see data/README.md).
%
% Outputs:
%   files  - N-by-1 cellstr of full paths to the .wav clips
%   labels - N-by-1 categorical species labels
%   groups - N-by-1 string of split groups (source recordings); pass to
%            split_data so clips of one recording stay on one side
%   info   - Manifest rows for the returned clips (XC IDs, recordists)

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(fileparts(mfilename('fullpath')), 'data');
end
if nargin < 2 || isempty(vocalization)
    vocalization = "song";
end
vocalization = lower(string(vocalization));

manifestFile = fullfile(dataDir, 'manifest.csv');
if ~isfile(manifestFile)
    error('load_dataset:noManifest', 'manifest.csv not found in %s', dataDir);
end

T = readtable(manifestFile, 'TextType', 'string', 'Delimiter', ',');

% Keep only the requested vocalization type
if vocalization ~= "all"
    T = T(T.vocalization == vocalization, :);
end
if isempty(T)
    error('load_dataset:empty', 'No "%s" clips listed in %s', ...
        vocalization, manifestFile);
end

% Build full paths and check that every clip exists
files = cellstr(fullfile(dataDir, T.file));
missing = ~cellfun(@isfile, files);
if any(missing)
    error('load_dataset:missingFiles', '%d clip(s) missing, e.g. %s', ...
        nnz(missing), files{find(missing, 1)});
end

labels = categorical(T.species);
groups = T.split_group;
info   = T;

fprintf('Loaded %d %s clips from %d recordings, %d species.\n', ...
    numel(files), vocalization, numel(unique(groups)), ...
    numel(categories(labels)));
end
