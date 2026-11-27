# Bird recordings dataset

Short WAV clips, each holding **one song** of a single species, cut from Xeno-canto recordings. Silence, other species and distorted audio were removed, and every clip was checked by eye on a spectrogram.

`manifest.csv` lists every clip with its label, source recording and citation details. **Load from the manifest, not by scanning folders.**

## Contents

| Species | Code | Songs (independent recordings) |
|---|---|---|
| American Redstart (*Setophaga ruticilla*) | AR | 35 (8) |
| Canada Warbler (*Cardellina canadensis*) | CW | 53 (8) |
| Common Yellowthroat (*Geothlypis trichas*) | CY | 31 (7) |
| Indigo Bunting (*Passerina cyanea*) | IB | 31 (8) |
| Northern Cardinal (*Cardinalis cardinalis*) | NC | 43 (8) |
| Northern Parula (*Setophaga americana*) | NP | 32 (9) |
| Painted Bunting (*Passerina ciris*) | PB | 43 (8) |
| Rose-breasted Grosbeak (*Pheucticus ludovicianus*) | RBG | 40 (8) |
| Scarlet Tanager (*Piranga olivacea*) | ST | 36 (8) |
| Yellow-breasted Chat (*Icteria virens*) | YBC | 44 (8) |

Independent recordings are `split_group`s: same-session recordings count once, so some species have more `Recording<N>` folders than this.

Most song clips hold one full song, 1.0–7.3 s long. Recordings added after the first batch contribute at most 6 clips each. American Redstart songs are short (0.8–1.5 s in Recording4–8).

**Yellow-breasted Chat clips are phrases, not whole songs.** The chat sings a slow string of unrelated phrases (rattles, whistles, "chack" notes) with 1–3 s gaps, so each clip is one phrase (0.5–3.6 s). They are shorter and more varied than the other species' clips.

## Layout

```
data/<Species>/song/Recording<N>/<CODE>_song_<n>.wav
```

- **The label is the species folder, two levels above the file.** `LabelSource='foldernames'` in `audioDatastore` would label clips "Recording1", "Recording2"… — use the manifest's `species` column instead.
- Each `Recording<N>` folder is one source recording. Clip numbers run continuously across a species' recordings (e.g. `IB_song_1`–`3` are Recording1, `IB_song_4`–`8` Recording2).
- Clips keep their source's sample rate (44.1 or 48 kHz). Nine Painted Bunting song clips are stereo. `preprocess_audio.m` converts everything to 22050 Hz mono.

## manifest.csv columns

| Column | Meaning |
|---|---|
| `file` | Path relative to `data/` |
| `species`, `scientific_name`, `species_code` | Label |
| `vocalization` | Always `song` |
| `recording` | `Recording<N>` folder |
| `split_group` | **Group to split train/test by.** Usually the Xeno-canto ID; recordings by the same recordist at the same place on the same day (likely the same bird) share one group, e.g. `XC79011+XC79012+XC79013` |
| `xc_id`, `recordist`, `location`, `date` | Source recording, for citations |
| `source_file`, `source_start_s`, `source_end_s` | Where the clip was cut from (file in `../source_recordings/`). Times are as decoded by ffmpeg; MATLAB's `audioread` can place MP3 audio up to ~0.05 s later because decoders handle the MP3 start-up padding differently |
| `duration_s`, `original_sample_rate`, `channels` | Audio format |
| `other_species_in_source` | Background species the recordist listed (removed from clips where audible) |

## Splitting train/test

Split **by `split_group`**, never by clip.

- Use `split_group`, not the folder name: seven sets of same-session recordings are merged (Canada Warbler has two; Common Yellowthroat, Northern Parula, Rose-breasted Grosbeak, Scarlet Tanager and Yellow-breasted Chat one each).
- Songs per species come from 7–9 independent groups. With that few, an 80/20 split leaves one or two recordings per species in the test set, so accuracy will swing between seeds. **Leave-one-recording-out cross-validation** gives a more reliable number.

```matlab
dataDir = 'data';
T = readtable(fullfile(dataDir, 'manifest.csv'), 'TextType', 'string');
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

- **Small:** 7–9 independent recordings per species (WORK_SPLIT targets 25–40). Ten species meets the proposal's 10–15. Fine for building and debugging the pipeline; add recordings before reporting final results.
- **Painted Bunting song Recording1 (`PB_song_1`–`5`)** lost its source file before the manifest was made. It was found again and matched to **XC314884** (Jim Holmes) by cross-correlating the clips against it. Jim Holmes recorded Indigo Bunting XC314892 and Northern Parula XC314880+ at the same site that morning, so these three groups share a microphone and background across different species.
- **Similar species:** American Redstart, Canada Warbler, Common Yellowthroat and Northern Parula are all wood-warblers singing at 3–7 kHz, so expect most confusions among them.
- **Background noise that couldn't be removed:**
  - Indigo Bunting Recording3 (`IB_song_9`–`12`) has a constant ~5.7 kHz tone and clicks.
  - Indigo Bunting Recording7 (`IB_song_23`–`26`) has another bird's chip notes every ~0.7 s.
  - Canada Warbler Recording5 (`CW_song_17`–`18`, XC436648) has heavy steady noise.
  - Northern Parula Recording7 (XC54777) and Common Yellowthroat Recording4 (XC54740) have broadband clicks in the source.
  - Yellow-breasted Chat Recording3 and Recording5 have a constant tone (insect or equipment).
  - If any of these hurt accuracy, try dropping that recording.
- **Yellow-breasted Chat XC13082** was identified by the recordist from comparison with XC13688 (bird not seen).
- **WAVs without embedded metadata:** for XC792359, XC804269, XC1082125, XC1145165, XC1187881 and XC1194351 the recordist, location, date and background species were filled in from their Xeno-canto pages (stored in `xc_metadata_overrides.csv`, which the manifest generator reads). None shares a recordist and date with another recording here, so none is a same-session duplicate. The same was done for XC893207, XC899265, XC1006790, XC1013411, XC1013554, XC1015018, XC1135113, XC1135116 and XC1193452, so every clip now has a recordist. XC1135113 and XC1135116 (Yellow-breasted Chat, Mario Reyes Jr, Hisle Park, 2026-05-16 09:00) are one session, so they share a `split_group`.
- **Background species that are also classes here:** American Redstart XC1194351 (`AR_song_30`–`35`) lists Northern Cardinal and Indigo Bunting, and Canada Warbler XC1193452 lists Common Yellowthroat. If those clips are often confused with either species, check them by ear.
- **Common Yellowthroat XC76937 and XC78067** (Stuart Fisher, Big Creek Trail, 2011-04-20, a few metres apart) are one session and share a `split_group`.
- **Rose-breasted Grosbeak XC134880** (`RBG_song_37`–`40`) has a Downy Woodpecker drumming in short bursts at about 1.8 kHz, partly under each song.
- **Scarlet Tanager XC1002530** (`ST_song_21`) has a faint American Toad trill under it; its second song was left out because a woodpecker drums over it.
- Painted Bunting songs from the first five recordings are capped at 25. The other 44 clips are in `../excluded_clips/`, with `renumber_map.csv` linking old and new names.

## Citation

All source recordings are from [xeno-canto.org](https://xeno-canto.org); cite each by its XC number and recordist (both in the manifest). Xeno-canto recordings are under Creative Commons licences that vary by recording, so check each recording's page for its licence terms.
