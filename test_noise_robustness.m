
function noiseResults = test_noise_robustness(model, testFiles, Ytest)
% TEST_NOISE_ROBUSTNESS Evaluate classification under white noise.
% Inputs:
%   model     - Trained classifier with normalization statistics
%   testFiles - Cell array of test recording paths
%   Ytest     - True species labels
%
% Output:
%   noiseResults - Structure containing SNR levels and accuracy

snrLevels = [20, 10, 0];
accuracies = zeros(size(snrLevels));

if numel(testFiles) ~= numel(Ytest)
    error('Test files and labels must have matching lengths.');
end

if isempty(testFiles)
    error('No test recordings were provided.');
end

for s = 1:numel(snrLevels)
    snr = snrLevels(s);
    Xnoisy = [];

    fprintf('Testing at %d dB SNR...\n', snr);

    for i = 1:numel(testFiles)
        % Load original test recording
        [audio, fs] = audioread(testFiles{i});

        % Add white Gaussian noise at the target SNR
        noisyAudio = awgn(audio, snr, 'measured');

        % Apply the same preprocessing as training
        [processed, fsOut] = preprocess_audio(noisyAudio, fs);

        % Extract features
        features = extract_features(processed, fsOut);
        Xnoisy(i, :) = features;
    end

    % Evaluate noisy recordings
    results = evaluate_model(model, Xnoisy, Ytest, false);
    accuracies(s) = results.acc;

    fprintf('Accuracy at %d dB: %.2f%%\n', ...
        snr, results.acc * 100);
end

% Plot accuracy against noise level
noiseResults.fig = figure;
plot(snrLevels, accuracies * 100, '-o', 'LineWidth', 2);
xlabel('Signal-to-Noise Ratio (dB)');
ylabel('Classification Accuracy (%)');
title('Bird Species Classification Under Noise');
grid on;
ylim([0 100]);
xticks(snrLevels);

noiseResults.snrLevels = snrLevels;
noiseResults.accuracies = accuracies;
end
