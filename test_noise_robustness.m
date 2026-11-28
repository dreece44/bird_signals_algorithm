function noiseResults = test_noise_robustness(model, testFiles, Ytest)
% TEST_NOISE_ROBUSTNESS Evaluate accuracy under repeatable Gaussian noise.
% Uses five independent noise trials per SNR, with the same random seed
% for each classifier call so SVM and ensemble see identical noisy clips.
% Outputs mean accuracies, per-trial accuracies, and standard deviations.

snrLevels = [20, 10, 0];
numTrials = 5;
trialAccuracies = zeros(numTrials, numel(snrLevels));

if numel(testFiles) ~= numel(Ytest)
    error('Test files and labels must have matching lengths.');
end
if isempty(testFiles)
    error('No test recordings were provided.');
end

% Keep this function from changing the caller's random-number state.
previousRng = rng;
restoreRng = onCleanup(@() rng(previousRng)); %#ok<NASGU>
rng(2026, 'twister');

for trial = 1:numTrials
    for s = 1:numel(snrLevels)
        snrDb = snrLevels(s);
        Xnoisy = [];

        fprintf('Noise trial %d/%d at %d dB SNR...\n', ...
            trial, numTrials, snrDb);

        for i = 1:numel(testFiles)
            [audio, fs] = audioread(testFiles{i});
            noisyAudio = awgn(audio, snrDb, 'measured');
            [processed, fsOut] = preprocess_audio(noisyAudio, fs);
            features = extract_features(processed, fsOut);
            Xnoisy(i, :) = features;
        end

        evaluation = evaluate_model(model, Xnoisy, Ytest, false);
        trialAccuracies(trial, s) = evaluation.acc;
    end
end

accuracies = mean(trialAccuracies, 1);
stdAccuracies = std(trialAccuracies, 0, 1);

for s = 1:numel(snrLevels)
    fprintf('%d dB: mean %.2f%%, SD %.2f percentage points (%d trials)\n', ...
        snrLevels(s), accuracies(s)*100, stdAccuracies(s)*100, numTrials);
end

% Display SNR from low to high.
[sortedSnr, order] = sort(snrLevels);
noiseResults.fig = figure;
errorbar(sortedSnr, accuracies(order)*100, ...
    stdAccuracies(order)*100, '-o', 'LineWidth', 2);
xlabel('Signal-to-Noise Ratio (dB)');
ylabel('Mean Classification Accuracy (%)');
title(sprintf('Bird Species Classification Under Noise (%d Trials)', numTrials));
grid on;
ylim([0 100]);
xticks(sortedSnr);

noiseResults.snrLevels = snrLevels;
noiseResults.accuracies = accuracies;
noiseResults.stdAccuracies = stdAccuracies;
noiseResults.trialAccuracies = trialAccuracies;
noiseResults.numTrials = numTrials;
noiseResults.seed = 2026;
end
