function plot_feature_figures(dataDir, outDir)
%PLOT_FEATURE_FIGURES  Report figures for the Feature Extraction section.
%   plot_feature_figures(dataDir, outDir)
%   Inputs:  dataDir - folder holding manifest.csv (default 'data')
%            outDir  - where PNGs are written (default 'figures')
%   Writes:  species_spectrograms.png - STFT spectrograms of 4 species, same axes
%            feature_boxplots.png     - 4 key features across all 10 species

if nargin < 1, dataDir = 'data'; end
if nargin < 2, outDir = 'figures'; end
if ~isfolder(outDir), mkdir(outDir); end
T = readtable(fullfile(dataDir, 'manifest.csv'), 'TextType', 'string');

% ---- Figure 1: spectrograms (same STFT settings as extract_features) ------
clips = ["NC_song_1" "IB_song_1" "NP_song_8" "CY_song_4"];
fig = figure('Position', [100 100 1000 650], 'Theme', 'light');
tl = tiledlayout(2, 2, 'TileSpacing', 'compact');
for k = 1:numel(clips)
    r = T(endsWith(T.file, "/" + clips(k) + ".wav"), :);
    [y, fs] = load_clip(fullfile(dataDir, r.file));
    [S, f, t] = spectrogram(y, hann(512, 'periodic'), 384, 512, fs);
    SdB = 10 * log10(abs(S).^2 + eps);
    nexttile;
    imagesc(t, f / 1000, SdB - max(SdB(:)), [-60 0]); axis xy;
    xlim([0 3]); ylim([0 11]); set(gca, 'Color', [0.19 0.07 0.23]);  % turbo's floor
    title(sprintf('%s (%s)', r.species, r.xc_id));
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
end
colormap(fig, 'turbo'); cb = colorbar; cb.Layout.Tile = 'east';
cb.Label.String = 'Power relative to peak (dB)';
title(tl, 'STFT spectrograms (512-pt Hann, 75% overlap, fs = 22.05 kHz)');
exportgraphics(fig, fullfile(outDir, 'species_spectrograms.png'), 'Resolution', 200);

% ---- Figure 2: how a few features separate all 10 species ----------------
X = zeros(height(T), 43);
for i = 1:height(T)
    [y, fs] = load_clip(fullfile(dataDir, T.file(i)));
    [X(i, :), names] = extract_features(y, fs);
end
species = categorical(T.species_code);
show  = ["peakFreq_mean" "bandwidth_mean" "entropy_mean" "fmRate"];
units = ["kHz" "kHz" "normalised" "kHz/s"];
fig = figure('Position', [100 100 1100 650], 'Theme', 'light');
tiledlayout(2, 2, 'TileSpacing', 'compact');
for k = 1:numel(show)
    nexttile;
    boxchart(species, X(:, names == show(k)));
    title(strrep(show(k), '_', ' ')); ylabel(units(k)); grid on;
end
exportgraphics(fig, fullfile(outDir, 'feature_boxplots.png'), 'Resolution', 200);
end

function [y, fs] = load_clip(file)
% Read one clip and preprocess it the same way the pipeline does.
[x, fsIn] = audioread(file);
if exist('preprocess_audio', 'file') == 2
    [y, fs] = preprocess_audio(x, fsIn);
else    % stand-in until preprocess_audio.m exists: mono, 22050 Hz, 1-10 kHz, normalised
    fs = 22050;
    y = resample(mean(x, 2), fs, fsIn);
    [b, a] = butter(4, [1000 10000] / (fs / 2), 'bandpass');
    y = filtfilt(b, a, y);  y = y / max(abs(y));
end
end
