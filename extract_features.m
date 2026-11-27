function [feat, names] = extract_features(y, fs)
%EXTRACT_FEATURES  Fixed-length DSP feature vector for one bird recording.
%   [feat, names] = extract_features(y, fs)
%   Inputs:  y     - preprocessed mono signal (vector), output of preprocess_audio
%            fs    - sample rate in Hz (22050 in this pipeline)
%   Outputs: feat  - 1x43 double row vector (same length for every recording)
%            names - 1x43 string array naming each entry of feat (optional)
%
%   Every feature is summarised by its mean and std over the STFT frames:
%     MFCC 1-13 (32 mel bands spanning 1-10 kHz)                   26
%     spectral centroid, bandwidth, 85% rolloff, peak frequency,
%       spectral entropy, spectral flatness (from the STFT)         12
%     zero-crossing rate, log short-time energy                      4
%   plus the frequency-modulation rate (median |d peak freq / dt|)   1

y = y(:);
winLen = 512;  hop = 128;          % 23 ms Hann window, 5.8 ms hop at 22050 Hz
band = [1000 10000];               % song band, same as the preprocessing band-pass
if numel(y) < winLen, y(end+1:winLen) = 0; end
win = hann(winLen, 'periodic');

% ---- STFT power spectrum, keeping only the song band ---------------------
[S, f] = stft(y, fs, Window=win, OverlapLength=winLen-hop, FFTLength=winLen, ...
              FrequencyRange='onesided');
inBand = f >= band(1) & f <= band(2);
P = abs(S(inBand, :)).^2;   f = f(inBand) / 1000;          % power, f in kHz
nFrames = size(P, 2);

% Frames within 40 dB of the loudest carry the song; the rest are gaps
frameE = sum(P, 1);
active = frameE >= 1e-4 * max(frameE);

% Dominant frequency per frame, refined between bins by parabolic interpolation
L = log(P + eps);
[~, k] = max(L, [], 1);
k = min(max(k, 2), numel(f) - 1);
idx = sub2ind(size(L), k, 1:nFrames);
a = L(idx - 1);  b = L(idx);  c = L(idx + 1);
delta = 0.5 * (a - c) ./ (a - 2*b + c - eps);          % offset in bins, |delta| <= 0.5
pkAll = f(k)' + max(min(delta, 0.5), -0.5) * (f(2) - f(1));
Pa = P(:, active);
p  = Pa ./ (sum(Pa, 1) + eps);      % each frame as a distribution over frequency

% ---- spectral shape (one value per active frame) --------------------------
centroid  = f' * p;                                       % kHz
bandwidth = sqrt(sum((f - centroid).^2 .* p, 1));          % kHz
[~, iRoll] = max(cumsum(p, 1) >= 0.85, [], 1);
rolloff   = f(iRoll)';                                     % kHz
peakF     = pkAll(active);                                 % dominant frequency, kHz
entropy   = -sum(p .* log2(p + eps), 1) / log2(numel(f));  % 0 = pure tone, 1 = noise
flatness  = exp(mean(log(Pa + eps), 1)) ./ (mean(Pa, 1) + eps);  % geo / arith mean

% Frequency modulation: how fast the dominant frequency glides inside a note.
% Jumps over 0.5 kHz between neighbouring frames are note changes, not glides.
dPeak  = abs(diff(pkAll));
pairs  = active(1:end-1) & active(2:end) & dPeak < 0.5;
fmRate = 0;
if any(pairs), fmRate = median(dPeak(pairs)) * fs / hop; end   % kHz per second

% ---- MFCCs on the same frames, mel filters restricted to the song band ----
edges = mel2hz(linspace(hz2mel(band(1)), hz2mel(band(2)), 34));
C = mfcc(y, fs, Window=win, OverlapLength=winLen-hop, BandEdges=edges, ...
         NumCoeffs=13, LogEnergy='ignore');
C = C(active, :);

% ---- time-domain features on the same frames ------------------------------
frames = y((1:winLen)' + (0:nFrames-1) * hop);
zcr  = mean(abs(diff(sign(frames(:, active)))) > 0, 1);   % crossings per sample
logE = 10 * log10(mean(frames.^2, 1) + eps);              % all frames: keeps rhythm

% ---- assemble -------------------------------
ms   = @(v) [mean(v), std(v)];
feat = [mean(C, 1), std(C, 0, 1), ms(centroid), ms(bandwidth), ms(rolloff), ...
        ms(peakF), ms(entropy), ms(flatness), ms(zcr), ms(logE), fmRate];

if nargout > 1
    k = string(1:13);
    spec = ["centroid" "bandwidth" "rolloff" "peakFreq" "entropy" "flatness" "zcr" "logEnergy"];
    names = ["mfcc" + k + "_mean", "mfcc" + k + "_std", ...
             reshape([spec + "_mean"; spec + "_std"], 1, []), "fmRate"];
end
end
