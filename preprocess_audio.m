
function [y, fs] = preprocess_audio(x, fsIn) % column vector, fs = 22050
% PREPROCESS_AUDIO Clean up the raw audio recording before extracting
% features
% Inputs:
%   x       - raw audio sample (NxC matrix, can have any number of
%   channels)
%   fsIn    - the sample rate of x (Hz)
%   
% Output:
%   y       - cleaned up mono signal (column vector)
%   fs      - the output sample rate (always 22050Hz)

fs = 22050;      %our sample rate
fLow = 1000;     %band-pass lower cutoff (Hz)
fHigh = 10000;   %band-pass upper cutoff (Hz)
order = 4;       %Butterworth order 
frameMs =25;     %frame length (ms)
threshold = 0.3; %threshold position from noise floor to peak (dB)
maxSec = 10;     %maximum amount of seconds of active audio
minSec = 0.5;    %if signal less than this value, entire signal kept

%mono, double, zero-mean
x = double(x);
if size(x,1) < size(x,2), x=x.';
    %makes samples run down rows
end
x = mean(x, 2);
x = x-mean(x);

%resample signal to 22050Hz
if fsIn ~= fs
    [p, q] = rat(fs/fsIn); %rat function approximates values as rational fractions
    x = resample(x, p, q);
end

%normalize the peak amplitude to 1
x = x/(max(abs(x)) + eps);

%band-pass filter
%the Butterworth filter provides a maximally flat frequency 
% response in the passband with no ripples.
[b, a] = butter(order, [fLow fHigh]/(fs/2), 'bandpass'); %butter function computes the Butterworth bandpass filter coefficients 
x = filtfilt(b, a, x); %apply zero-phase filtering without phase shift

%energy-based activity detection on non-overlapping frames
frameLen = round(frameMs/1000*fs);
numFrames = floor(length(x)/frameLen);

if numFrames < 1
    y = x/(max(abs(x))+eps);
    return
end

frames = reshape(x(1:numFrames*frameLen), frameLen, numFrames);
Edb = 10*log10(sum(frames.^2, 1)/frameLen+eps);

noiseFloor = prctile(Edb, 10); %prctile function calculates the background noise floor using the 10th percentile of frame energies
noiseThreshold = noiseFloor+threshold*(max(Edb)-noiseFloor);
active = Edb>noiseThreshold;

y = frames(:, active);
y = y(:);

%if signal is too small, use the full filtered signal
if lenth(y) < minSec*fs
    y=x;
end

%limit length to keep feature extraction fast
y = y(1:min(end, round(maxSec*fs)));

%peak normalization
y = y/(max(abs(y)) + eps);
end