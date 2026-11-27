function [label, score] = predict_species(model, audioFile)

if ~isfile(audioFile)
    error('Audio file not found: %s', audioFile);
end

[x, fsIn] = audioread(audioFile);

[y, fs] = preprocess_audio(x, fsIn);

features = extract_features(y, fs);

features = features(:)';

if numel(features) ~= numel(model.mu)
    error('Feature count does not match training data.');
end

sigma = model.sigma;
sigma(sigma == 0) = 1;

features = (features - model.mu) ./ sigma;

if any(~isfinite(features))
    error('Extracted features contain invalid values.');
end

[predictedLabel, classScores] = predict( ...
    model.clf, features);

label = categorical(string(predictedLabel));

score = max(classScores, [], 2);

fprintf('Predicted species: %s\n', string(label));
fprintf('Classification score: %.4f\n', score);

end
