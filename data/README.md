# Bird recordings dataset

Short WAV clips, each holding **one song or one call** of a single species, cut from Xeno-canto recordings. Silence, other species and distorted audio were removed, and every clip was checked by eye on a spectrogram.

`manifest.csv` lists every clip with its label, source recording and citation details. **Load from the manifest, not by scanning folders.**

## Contents

| Species | Code | Songs (recordings) | Calls (recordings) |
|---|---|---|---|
| Indigo Bunting (*Passerina cyanea*) | IB | 31 (8) | none |
| Northern Cardinal (*Cardinalis cardinalis*) | NC | 25 (5) | 36 (2) |
| Painted Bunting (*Passerina ciris*) | PB | 25 (5) | 23 (2) |
| Rose-breasted Grosbeak (*Pheucticus ludovicianus*) | RBG | 23 (5) | 4 (1) |
| Scarlet Tanager (*Piranga olivacea*) | ST | 14 (3) | 17 (2) |

Songs are 1.4–5.8 s long (1.36 s minimum). Call clips have a fixed length: 0.45 s (PB, NC, RBG) or 0.9 s (ST, whose "chip-burr" call is longer).

**Use songs only for the species classifier.** Every species has songs from several recordings. Calls are uneven: Indigo Bunting has none, and Rose-breasted Grosbeak has 4 from one recording. Mixing them in would let the model learn "call vs. song" instead of species.

## Layout

```
data/<Species>/<song|call>/Recording<N>/<CODE>_<song|call>_<n>.wav
```

- **The label is the species folder, two levels above the file.** `LabelSource='foldernames'` in `audioDatastore` would label clips "Recording1", "Recording2"… — use the manifest's `species` column instead.
- Each `Recording<N>` folder is one source recording. Clip numbers run continuously across a species' recordings (e.g. `IB_song_1`–`3` are Recording1, `IB_song_4`–`8` Recording2).
- Clips keep their source's sample rate (44.1 or 48 kHz). Nine Painted Bunting song clips are stereo. `preprocess_audio.m` converts everything to 22050 Hz mono.

## manifest.csv columns

| Column | Meaning |
|---|---|
| `file` | Path relative to `data/` |
| `species`, `scientific_name`, `species_code` | Label |
| `vocalization` | `song` or `call` |
| `recording` | `Recording<N>` folder |
| `split_group` | **Group to split train/test by** (the Xeno-canto ID) |
| `xc_id`, `recordist`, `location`, `date` | Source recording, for citations |
| `source_file`, `source_start_s`, `source_end_s` | Where the clip was cut from (file in `../source_recordings/`) |
| `duration_s`, `original_sample_rate`, `channels` | Audio format |
| `other_species_in_source` | Background species the recordist listed (removed from clips where audible) |

## Splitting train/test

Split **by `split_group`**, never by clip. Clips from one recording share the same bird, microphone and background, so putting them on both sides inflates test accuracy.

- Use `split_group`, not the folder name: **XC361434 is both Painted Bunting `song/Recording5` and `call/Recording2`**.
- With only 3–8 recordings per species, an 80/20 split leaves about one recording per species in the test set, so accuracy will swing between seeds. **Leave-one-recording-out cross-validation** gives a more reliable number.

```matlab
dataDir = 'data';
T = readtable(fullfile(dataDir, 'manifest.csv'), 'TextType', 'string');
T = T(T.vocalization == "song", :);            % species classifier: songs only
files  = fullfile(dataDir, T.file);
labels = categorical(T.species);
groups = T.split_group;

% 80/20 split by recording, stratified by species
rng(42);
testIdx = false(height(T), 1);
for s = categories(labels)'
    g = unique(groups(labels == s{1}));
    g = g(randperm(numel(g)));
    nTest = max(1, round(0.2 * numel(g)));
    testIdx = testIdx | (labels == s{1} & ismember(groups, g(1:nTest)));
end
trainIdx = ~testIdx;
```

## Known limitations

- **Small:** 3–8 recordings per species (WORK_SPLIT targets 25–40) and 5 species (the proposal says 10–15). Fine for building and debugging the pipeline; add recordings before reporting final results.
- **Painted Bunting song Recording1 (`PB_song_1`–`5`) has no source file.** It was deleted before the manifest was made, so its Xeno-canto ID is unknown. It has its own `split_group` so it still splits correctly, but it can't be cited — replace or drop it for the final report.
- **Northern Cardinal song Recording5 (XC792359)** came as a WAV with no embedded metadata; look up the recordist and location on its Xeno-canto page.
- **Background noise that couldn't be removed:**
  - Indigo Bunting Recording3 (`IB_song_9`–`12`) has a constant ~5.7 kHz tone and clicks.
  - Indigo Bunting Recording7 (`IB_song_23`–`26`) has another bird's chip notes every ~0.7 s.
  - If either hurts accuracy, try dropping that recording.
- Painted Bunting songs are capped at 25, spread across all five recordings. The other 44 clips are in `../excluded_clips/`, with `renumber_map.csv` linking old and new names.

## Other folders (outside `data/`)

- `../source_recordings/` — the original Xeno-canto files, in matching `Recording<N>` folders. Subfolders named `excluded_*` or `no_calls_found` hold recordings that were deliberately not used:
  - Scarlet Tanager XC415181: the recordist thought it might be a vireo.
  - Scarlet Tanager XC54750: every song overlaps Indigo Bunting.
  - Northern Cardinal XC236565: the recording is clipped (distorted).
  - Northern Cardinal XC792359 call copy: it contains only songs.
- `../excluded_clips/` — usable clips left out to balance classes.

## Citation

All source recordings are from [xeno-canto.org](https://xeno-canto.org); cite each by its XC number and recordist (both in the manifest). Xeno-canto recordings are under Creative Commons licences that vary by recording, so check each recording's page for its licence terms.
