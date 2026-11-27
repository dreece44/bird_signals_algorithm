# ALGORITHM → B. Feature Extraction  (Jack Anthony)

*Drop-in section for the paper template. Figures are in `figures/` and are produced by `plot_feature_figures.m`. Equations are in LaTeX. Word's equation editor (Alt+=) accepts them pasted as-is.*

---

A classifier needs every recording to be described by the same number of values, but the clips in our dataset range from 0.5 s to 7.3 s. `extract_features.m` turns one preprocessed clip, sampled at 22 050 Hz, into a fixed 43-element feature vector. It does this in three steps. First it computes a short-time Fourier transform (STFT) of the clip. Next it measures the shape of the spectrum, the cepstrum and the waveform in every short frame. Finally it summarises each frame-level measurement by its mean and standard deviation over the clip. The mean describes the song's typical sound, and the standard deviation describes how much that sound changes from note to note.

### 1) Short-time Fourier transform

A bird song is non-stationary: its pitch can sweep several kilohertz in under 100 ms (Fig. X1). One FFT of the whole clip would blur these sweeps together, so we use the STFT:

$$X[k,m] = \sum_{n=0}^{N-1} y[n + mH]\, w[n]\, e^{-j2\pi kn/N}$$

Here $w[n]$ is an $N = 512$-point periodic Hann window (23.2 ms) and $H = 128$ is the hop (5.8 ms, 75 % overlap). This window gives a frequency resolution of $f_s/N = 43$ Hz, fine enough to separate notes a few hundred hertz apart. It is still short enough to follow fast trills such as the Northern Parula's buzz, whose notes are about 60 ms apart. The Hann window keeps spectral leakage from loud notes from spilling into neighbouring bins.

We keep only bins between 1 and 10 kHz, the same band as the preprocessing band-pass filter, and work with the power spectrum $P[k,m] = |X[k,m]|^2$. Even after preprocessing, a clip contains short gaps between notes. A frame is treated as *active* if its in-band energy is within 40 dB of the loudest frame. Spectral-shape features are computed only on active frames, so near-silent gaps do not pull the averages towards the noise floor.

### 2) Spectral-shape features

For each active frame, the power spectrum is normalised into a distribution over frequency, $p[k] = P[k] / \sum_k P[k]$. Six features are then computed from it:

| Feature | Definition | What it captures |
|---|---|---|
| Spectral centroid | $\mu = \sum_k f_k\, p[k]$ | "Centre of mass" of the spectrum (overall pitch) |
| Spectral bandwidth | $\sigma = \sqrt{\sum_k (f_k-\mu)^2 p[k]}$ | Pure whistle (narrow) vs. buzzy or harmonic note (wide) |
| Spectral roll-off | smallest $f_r$ with $\sum_{f_k \le f_r} p[k] \ge 0.85$ | Upper edge of most of the energy |
| Peak (dominant) frequency | $f_{pk} = \arg\max_k P[k]$, refined by parabolic interpolation | Pitch of the loudest component |
| Spectral entropy | $H = -\sum_k p[k]\log_2 p[k] \,/\, \log_2 K$ | 0 for a pure tone, 1 for white noise |
| Spectral flatness | $\left(\prod_k P[k]\right)^{1/K} / \left(\frac{1}{K}\sum_k P[k]\right)$ | Tonal vs. noise-like, using the geometric-to-arithmetic mean ratio |

The 43 Hz bin spacing limits how precisely the peak frequency can be measured. To get sub-bin precision, we fit a parabola through the log power of the peak bin and its two neighbours ($\alpha, \beta, \gamma$) and take its vertex [8]:

$$\delta = \tfrac{1}{2}\,\frac{\alpha-\gamma}{\alpha - 2\beta + \gamma}, \qquad \hat f_{pk} = f_{k^\*} + \delta\,\Delta f, \quad |\delta|\le \tfrac12 .$$

The refined peak track is also used for a **frequency-modulation rate**: the median of $|\hat f_{pk}[m+1]-\hat f_{pk}[m]|/(H/f_s)$ over consecutive active frames, in kHz/s. Steps larger than 0.5 kHz between frames are treated as jumps to a new note, not glides, and are ignored. This feature separates slurred, sweeping songs (Common Yellowthroat, American Redstart) from steadier ones (Scarlet Tanager, Rose-breasted Grosbeak). Without interpolation it could only take multiples of 7.4 kHz/s (one bin per hop), which made it almost useless.

### 3) Mel-frequency cepstral coefficients (MFCCs)

MFCCs [3], [6] summarise the *envelope* of the spectrum in a few decorrelated numbers. MATLAB's `mfcc` computes them by applying a bank of triangular filters spaced evenly on the mel scale, $m = 2595\log_{10}(1 + f/700)$, to the STFT power. It then takes the logarithm of each filter's energy and applies a discrete cosine transform (DCT). We keep the first 13 coefficients.

The default filter bank was designed for human speech: most of its filters lie below 1 kHz, where our band-passed signals contain nothing. We therefore pass `mfcc` our own band edges: 32 filters spaced evenly on the mel scale **between 1 and 10 kHz**. This puts all the frequency resolution where the songs actually are. The MFCCs use the same window, hop and active frames as the STFT features, so every feature describes exactly the same frames. The log-energy term is left out because loudness depends on recording distance, not species.

### 4) Time-domain features

Two simple waveform features are computed on the same frames. The **zero-crossing rate** is the fraction of adjacent samples that change sign; it is a cheap pitch and noisiness indicator. The **log short-time energy** is $10\log_{10}\big(\tfrac1N\sum_n y^2[n]\big)$. Its standard deviation is taken over *all* frames, including gaps, so it captures the song's rhythm: a song of separate notes has a much larger energy spread than a continuous trill (median 11.2 dB for Indigo Bunting vs. 8.6 dB for Northern Parula).

### 5) Feature vector

| Group | Features | Count |
|---|---|---|
| MFCC | 13 coefficients × (mean, std) | 26 |
| Spectral shape | centroid, bandwidth, roll-off, peak frequency, entropy, flatness × (mean, std) | 12 |
| Time domain | zero-crossing rate, log energy × (mean, std) | 4 |
| Modulation | frequency-modulation rate | 1 |
| **Total** | | **43** |

Features are in kHz and natural units. The classifier (Section C) z-score-normalises every column using training-set statistics only. Extraction takes about 16 ms per clip.

### 6) Why these features separate species

Fig. X1 shows that the four species occupy different parts of the time–frequency plane. The Northern Cardinal sings a series of loud, pure downward slurs from about 6 to 2 kHz, with most of the energy near 2–3 kHz. The Indigo Bunting sings paired notes that jump between 3 and 7 kHz. The Northern Parula sings a rising buzz of fast notes centred near 6 kHz. The Common Yellowthroat repeats a "witchety" phrase of up-and-down sweeps around 4–5 kHz.

Fig. X2 shows four of the features over all 388 clips. **Peak frequency** alone splits the species into a low group (Northern Cardinal 2.7 kHz, Rose-breasted Grosbeak 2.6 kHz, Scarlet Tanager 2.9 kHz, Yellow-breasted Chat 3.0 kHz; medians) and a high group (American Redstart 5.6 kHz, Northern Parula 5.8 kHz). Within a group, the other features break the ties. Scarlet Tanager has the narrowest bandwidth (0.36 kHz), and the Rose-breasted Grosbeak's clear, robin-like whistles give the lowest entropy (0.40). The yellowthroat's sweeping phrases give the highest modulation rate (10.6 kHz/s), while the tanager and grosbeak are steadiest (about 5 kHz/s). The Yellow-breasted Chat shows the widest spread on every feature, which matches its varied repertoire of rattles, whistles and "chack" notes. The warblers (American Redstart, Canada Warbler, Common Yellowthroat, Northern Parula) overlap on most features, which is where we expect the classifier to make most of its mistakes.

**Preliminary check (to be replaced by the final results in Section D).** As a quick check, we trained an untuned RBF-SVM using leave-one-recording-out cross-validation: each Xeno-canto recording group was held out in turn, so no clip from a test recording was ever seen in training. Chance level for 10 species is 10 %.

| Feature subset | Accuracy |
|---|---|
| MFCC only (26) | 65.5 % |
| Spectral + time-domain only (16) | 60.1 % |
| MFCC means + spectral + time-domain + FM (30) | 70.6 % |
| All 43 features | **71.4 %** |

The two feature families are complementary: either alone does worse than both together. Most remaining errors are between warblers, for example Northern Parula clips predicted as American Redstart. *(These numbers used a stand-in for `preprocess_audio.m`: mono, 22 050 Hz resampling, a 4th-order 1–10 kHz Butterworth band-pass and peak normalisation. They will change slightly with the final preprocessing and the tuned classifier.)*

---

**Figure captions**

- **Fig. X1** (`figures/species_spectrograms.png`). STFT spectrograms of one song each from Northern Cardinal (XC120865), Indigo Bunting (XC192312), Northern Parula (XC44333) and Common Yellowthroat (XC13687). The settings are the same as in feature extraction (512-point Hann window, 75 % overlap, $f_s$ = 22.05 kHz), and power is in dB relative to each clip's peak.
- **Fig. X2** (`figures/feature_boxplots.png`). Distribution of four features across all 388 clips of the 10 species: mean peak frequency, mean spectral bandwidth, mean spectral entropy and frequency-modulation rate. AR = American Redstart, CW = Canada Warbler, CY = Common Yellowthroat, IB = Indigo Bunting, NC = Northern Cardinal, NP = Northern Parula, PB = Painted Bunting, RBG = Rose-breasted Grosbeak, ST = Scarlet Tanager, YBC = Yellow-breasted Chat.

**References to add** (continuing the template's numbering)

[6] S. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, Aug. 1980.

[7] G. Peeters, "A large set of audio features for sound description (similarity and classification) in the CUIDADO project," IRCAM, Paris, Tech. Rep., 2004.

[8] J. O. Smith III, *Spectral Audio Signal Processing*, W3K Publishing, 2011, "Quadratic Interpolation of Spectral Peaks." [Online]. Available: https://ccrma.stanford.edu/~jos/sasp/

*(Cite [7] after "Spectral-shape features" for the centroid, bandwidth, roll-off, entropy and flatness definitions.)*
