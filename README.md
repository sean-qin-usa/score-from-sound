# score-from-sound

Turn a recording into a draft score. The tools here split a song into stems, write down the notes in each one, and export MIDI and MusicXML that open in MuseScore, so that arranging or learning a piece, whether for a string duo, a quartet, a piano or a vocal group, starts from a draft instead of a blank page.

There are two paths, because two kinds of recording fail in different ways. A pop song can be pulled apart by timbre (voice, bass, drums, everything else) and each stem transcribed on its own. A duet of two violins cannot, since the separation models rely on the instruments sounding different; there the notes are transcribed from the mix and divided into two parts afterwards.

## How well it works

The duet path is scored against nine two-violin recordings from the [URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html), which come with hand-checked annotations for every note of every player: 4,318 notes in total.

| | |
|---|---|
| Real notes found | **89.9%** |
| Written notes that are real | **88.1%** |
| Found notes placed in the right player's part | **84.2%** |
| Fast notes (under 0.15 s) found | 81.6% |
| Upper note found when the violins play in octaves | 73.4% |
| Notes 10 dB or more quieter than the other violin found | 84.5% |

The detector tends to write one held note as two or three, and on these pieces a back-to-back pair of the same pitch is a real repeated note about as often as it is a continuation, so merging them blindly costs 17 points of notes found. The recording settles it instead: at a real attack the level of that pitch jumps by a median of 6.8 dB and dips by 6.3 dB first, against 1.9 and -0.1 for a continuation, and a note is joined to the one before it only when neither shows. That check took the share of written notes that are real from 74.7% to 88.1% for 2.4 points of notes found; its two thresholds were chosen on half of the pieces, so part of that gain is in-sample (the held-out halves gave 13 and 15 points). What remains among the extra notes is mostly overtones an octave or a twelfth above a real note.

## A model that names the instrument

The built-in detector writes notes without saying what played them, which is why a piano swamps the violin parts. [YourMT3+](https://github.com/mimbres/YourMT3) (Chang et al., 2024; Apache-2.0, open weights) is a transcription model trained across many instrumentations that writes one track per instrument, and `scripts/run_yourmt3.py` runs it so that `scripts/duet.py --notes-from` can take its strings track as the notes to divide between the two players, carrying everything else along as an accompaniment part.

| | Built-in detector | YourMT3+ |
|---|---|---|
| Two violins and piano, checked against the score: share of violin-part notes that are violin notes | 18% | **61%** |
| Same piece: violin notes found, above the chance level | 43 points | 47 points |
| Same piece: violin notes placed in the right part, after the split by side | 83% | **86.5%** |
| Two-violin benchmark: written notes that are real | 87.5% | **96.8%** |
| Two-violin benchmark: upper note found when the violins play in octaves | 73.9% | 86.3% |

The benchmark row flatters the model, since URMP is among the data it was trained on; the piece with piano is one it had never seen, and is the fairer evidence. It still missed about four violin notes in ten there, it puts both violins on a single strings track (so the split remains this project's job), and on a reverberant two-violin recording with no answer key about half of its notes were the same pitch written again, most of them without a fresh attack in the audio, with 152 notes credited to a piano and a saxophone that were not there. `--join-held` applies the held-note check to its notes; it is off by default because it cost about two points of notes found on both recordings that have an answer key. `--lowest G3` moves anything below the instruments' range out of their parts.

```bash
.venv_mt3/bin/python scripts/run_yourmt3.py recording.wav out/my_piece/model.mid
python -I scripts/duet.py recording.wav out/my_piece --notes-from out/my_piece/model.mid --lowest G3
python -I scripts/duet.py recording.wav out/my_piece --notes-from out/my_piece/model.mid --lowest G3 --cross-check   # mark notes only one detector wrote
```

The model listens in windows of about two seconds, and a note that crosses the edge of a window is often cut in two, started late or lost. `run_yourmt3.py` therefore transcribes each recording twice, the second time with the window edges moved by half a window, and joins the two: a note both passes wrote is taken from the pass whose window edge is farther from it, and a note only one pass wrote is kept when it lies farther from that pass's edges than from the other's. Nothing in that rule was tuned. On the six recordings with a score it raised recall and precision together on every one, F from 74.4 to 75.7 on average and most on the Mozart quartet (82.0 to 84.2) and the Telemann (75.9 to 79.1), for twice the running time (`--single` runs one pass). A learned rule that kept or dropped each note from both note sources by its loudness, length, overtone relations and the like was also tried, trained on five recordings and tested on the sixth: it moved F from 72.2 to 72.6 on average, up on one recording and down on another, and was not adopted.

The model's code and checkpoint are not part of this repository: they go under `tools/yourmt3/`, with their own environment in `.venv_mt3`.

## Six recordings it had never heard

The benchmark above is studio audio that the instrument-aware model was trained on, so the fairer test is recordings from outside any training set, checked against scores that are already machine-readable (the music21 corpus and the Mutopia Project), with no page reading involved. `scripts/eval_ensemble.py` lines the score up with the recording, finds the transposition if the edition or the tuning differs, and counts.

| Recording | Notes found (recall): model | plain detector | Written notes in the score (precision): model | plain detector | Found notes given to the right player |
|---|---|---|---|---|---|
| Bach, Prelude in C, BWV 846 (piano) | **98%** | 88% | **95%** | 60% | one player |
| Mozart, quartet K. 458, Adagio | **90%** | 75% | 78% | 76% | 83% of four |
| Joplin, Maple Leaf Rag (piano) | **91%** | 80% | **83%** | 72% | one player |
| Telemann, canonic sonata no. 1, Vivace (two violins) | 81% | **83%** | 76% | **82%** | 80% with the canon rule, 50% without |
| Beethoven, quartet Op. 18 No. 1, Adagio (live) | 50% | 42% | 42% | 44% | 71% of four |
| Bach, chorale from the motet BWV 227 (choir; one pass of the model) | 43% | 53% | 42% | 40% | 78 to 85% of four |

Between 6% and 32% of notes would count as found by coincidence, most on the rag, whose sections repeat. Three things stand out. The instrument-aware model is not uniformly better: on two violins alone the plain detector with the held-note check matches or beats it. A live quartet in a slow movement is far harder than a studio one, and even ignoring fine timing only 68% of its pitches turn up in the right five-second stretch, against 89% for the Mozart. And voices are the weakest case of all: the model hears a choir mostly as strings and finds two notes in five, the plain detector finds the sopranos (81%) and loses the tenors (32%).

Dividing four players uses the same evidence as dividing two, plus the lowest note each instrument can produce (`split_voices` in `scripts/duet.py`); it beats stacking the notes top to bottom on the quartets and loses to it on the chorale, where all four voices move together. A canon defeats both, because the two players share one register and one line; there the players are told apart by the delay between them, found from pairs of notes that come back a fixed time later (`--canon`). Ordinary accompaniment figures repeat just as strongly at short delays, so that rule is an option and not automatic. Left/right position, which settled the two violins with piano, adds about a point on these quartets: the four players sit too close together in the stereo image.

### When the two detectors agree

The plain detector and the instrument-aware model are different networks trained on different recordings, so the working guess was that they seldom invent the same note. `scripts/eval_detectors.py` tests it by pairing their notes one to one (same pitch, starts within 0.15 s) and scoring each group against the score.

| Recording | Precision when both wrote the note | Only the model wrote it | Only the plain detector wrote it | Recall of the agreed notes alone | Recall of everything either wrote |
|---|---|---|---|---|---|
| Mozart quartet | **95%** | 73% | 46% | 64% | 95% |
| Bach prelude (piano) | 95% | 93% | 11% | 85% | 98% |
| Telemann (two violins) | **91%** | 59% | 70% | 71% | 91% |
| Joplin rag (piano) | 84% | 86% | 45% | 78% | 92% |
| Bach chorale (choir) | **72%** | 38% | 38% | 13% | 71% |
| Beethoven quartet (live) | 58% | 36% | 31% | 30% | 57% |

Agreement is the best single sign that a note is real on every one of the six, by 21 to 34 points over either source alone wherever strings or voices are involved, and it asks nothing about the instruments. On a piano the model needs no help: what only the plain detector wrote there is mostly overtones. `duet.py --cross-check` puts this to use. Notes only one source wrote are kept and coloured blue in the score, so the page shows which notes to check by ear. When the model hears one family almost alone (nine notes in ten), the notes only the plain detector wrote are added as well, since nothing else is playing that they could belong to; on the Telemann duet that lifts the notes found from 81% to 87%, with the doubtful 38% marked. With a piano in the room the plain detector's extra notes cannot be told from piano notes (8% of them are violin notes on the piece with piano), so there the rule only marks. Counting a note as agreed when the other source merely has that pitch sounding was tried and dropped: it mostly confirms the extra pieces of a held note that one source wrote in fragments.

The score is laid over the stretch of the recording where something is heard, from the first note to the end of the last, not over the whole file. Pinning the score's first note to the file's first second had shifted every opening that starts with silence (0.8 s on the piece with piano, where the first seconds of piano then looked missing), and correcting it raised the notes found on all eight recordings with a score, most on the rag (84% to 91%) and the prelude (93% to 97%); the right-player figures above were measured before this correction.

Lining the score up by individual notes instead of by harmony cut the typical timing error on the benchmark from 47 ms to 27 ms, and slow movements needed the score stretched to the recording's length first, since the alignment cannot bend the tempo by more than a factor of two.

## Beats and bar lines

Writing a recording as sheet music means finding where every beat falls, however much the tempo moves, and where each bar begins; the notes are then placed between the beats around them. The scores above give the true beat times of six recordings, so both beat finders were checked against them, in the score's own meter. "In step" is the share of the piece where the found beats keep matching the true ones, at the notated beat or at twice or half its rate.

| Recording | Meter | Old beat finder: in step | Beat This!: in step | Beat This!: bar lines right | Beat This!: beats per bar |
|---|---|---|---|---|---|
| Bach prelude (piano) | 4/4 | 4% | **59%** | 0.71 | 4, right |
| Joplin rag (piano) | 2/4 | 76% | 73% | 0.82 | 2, right |
| Mozart quartet, Adagio | 4/4 | 0% | **43%** | 0.35 | 2 (half bars) |
| Telemann, two violins, counted in quarters | 6/4 | **83%** | 32% | 0.35 | wrong |
| Beethoven quartet, live Adagio, counted in eighths | 9/8 | 31% | 18% | 0.07 | wrong |
| Two violins and piano (the score read from pages) | 4/4 | 19% | 21% | 0.29 | 2 (half bars) |

[Beat This!](https://github.com/CPJKU/beat_this) (Foscarin, Schlüter and Widmer, 2024; MIT) is a beat and bar tracker trained across many styles; `scripts/run_beats.py` runs it from `tools/beat_this/` in the instrument-aware model's environment. It settles the piano pieces, meter included, and fails where the old finder fails: slow, freely played strings. Neither is reliable enough yet to put bar lines into every score unattended, so the scores written here are drafts whose bar lines need checking wherever the music is slow or free.

## Where it falls short

Telling the two players apart is the weak point, and what helps depends on the recording. In mono, nothing in the pitches says which violin is playing when one of them plays alone: on a piece with a full score to check against, the two violins' solo passages sat in practically the same range, and the best possible pitch cut-off placed 57% of those notes correctly. A stereo recording can carry the answer in where each player sits. When the notes of a recording are spread widely between left and right, the parts are divided by side instead, which on that piece (two violins about 7 dB apart, with piano) placed 83% of violin notes in the right part against 62% for pitch movement; mixing the two kinds of evidence did worse than position alone, because the piano's notes pull the pitch-based logic around. The spread that triggers this was set from three recordings and `--no-stereo` turns it off. For mono recordings a `--solo` hint for passages you know are one player, and a `--roles` mode that divides the music into a sustained line and a moving line, are the workarounds; the roles mode suits pieces where each player keeps a role for a whole phrase and scored worse than the plain split on the piece above, where the pattern changes almost every bar. A relative version was tried and dropped: each part remembered the typical length of its recent notes, so that a note went to the part whose recent notes it resembled and roles could change only after a rest. It moved the right-player share on the two-violin benchmark from 85.0% to 85.7%, lowered it from 74.6% to 72.0% on the piece with piano when split by pitch alone, and did not make the roles on the mono duet any steadier.

Where the parts go wrong was measured on Smoke Gets in Your Eyes, the one two-violin piece with an answer key by part, using the instrument-aware model's notes:

| When | Split by position and line (the recording is stereo) | Split by line only (as for a mono recording) |
|---|---|---|
| One violin alone | 92% | 55% |
| Both play, in their usual order | 95% | 92% |
| Both play in octaves | 73% | 73% |
| Both play in unison | 62% | 62% |

In stereo nearly every error was a single note dropped into the wrong part, after which the line carried on where it belonged. So a lone note no longer interrupts a line: a note with nothing else of its part within a second, while the other part's line runs on both sides of it and that part is free at that moment, is moved into the other part (`heal_lines` in `scripts/duet.py`). That raised the share of violin notes in the right part from 89.9% to 91.9% and cut the breaks in the second violin's line from 24 to 16, and left the two-violin benchmark unchanged (85.0% and 84.9%). A first version that let a note move into a part still sounding for up to half its length made false chords and was tightened to the splitter's own rule, that two notes overlapping by more than 0.15 s are two notes at once. In mono the remaining problem is different: when one violin plays alone nothing in a mono recording says which, and the split is right only about half the time there. Five ways of deciding it were tested on the answer keys and none worked. Of the musical rules (the player who just played carries on, the player who plays next, the player whose register is nearer, solos alternate), the best (the other player takes over) was right for 18 of 28 solo passages. The tone of the two violins, averaged over the whole passage and learned from the stretches where both play, placed 54% of solo notes right on Smoke Gets in Your Eyes against 56% before, and lowered the benchmark from 84.9% to 84.1%. Writing each solo passage whole in the part that already held most of it made the lines read better but lowered accuracy on both. So this is left to someone who knows the piece: `--solo 0-13:1,95-102:2` gives time ranges where one violin plays alone and, after the colon, which one, and these hints are applied after every automatic rule so that none undoes them. A unison is the other weak spot, since the transcription has one note where two players play it; writing such notes into both parts is the next rule to try.

Chords inside a single part are kept when the notes are held together, but the benchmark players never play double stops, so that behaviour is unmeasured. Recordings with a piano remain a poor fit for the duet path, because the piano's notes end up in the violin parts: on the piece above fewer than one written note in five is a violin note. Telling struck notes from bowed ones note by note reached 69% against 59% for guessing, which is not enough to filter on. `--check-overtones` removes notes the recording says are only an overtone of a lower note, a small gain with piano and a net loss on two violins alone, where it costs about five points on the upper note of an octave. Notes inside a chord are found about half as often as single notes.

The pop path has no ground truth behind it yet. Sung melodies and bass lines come out as single lines with a confidence score per note; spoken or rapped passages are flagged as unclear and drawn with x noteheads, since their pitches are guesses. Off-scale notes are moved a semitone onto the song's scale unless `--no-autotune` is given. Chord roots are stable from bar to bar, while major versus minor is not reliable.

## Running it

Python 3.10 and a GPU with about 8 GB of memory are enough. The separation models download themselves on first use and take a few gigabytes. Fetching a recording from YouTube with yt-dlp now also needs a JavaScript runtime (Node.js or Deno) and the `yt-dlp-ejs` package, run as `yt-dlp --js-runtimes node -x --audio-format wav <url>`; without them YouTube refuses the download.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

For a pop song, starting from an audio file you have the right to use:

```bash
scripts/run_pop.sh path/to/song.wav my_song
```

This writes `out/my_song/` with the stems, a tempo-aligned MIDI file, a MusicXML score (lead, backing vocals, bass), a chord chart and a report. For two violins:

```bash
python -I scripts/duet.py path/to/duet.wav out/my_duet
python -I scripts/duet.py path/to/duet.wav out/my_duet --solo 0-13:2   # the second violin alone in the first 13 s
python -I scripts/duet.py path/to/duet.wav out/my_duet --roles         # sustained line / moving line
```

## Checking by ear

Every run writes `check_*.wav` files with the recording in the left ear and the transcription, played as plain tones, in the right. For duets `check_parts.wav` puts one part in each ear, which is the quickest way to hear where the split goes wrong.

## Checking against a score

`scripts/eval_urmp.py` reproduces the table above (the `merge_held` variant is the default pipeline) from a local copy of the URMP pieces that contain two violins, and reports the same figures broken down by note length, loudness and interval, which is how each setting was chosen. `scripts/eval_score.py` compares a transcription with sheet music in MusicXML or MIDI by stretching the score onto the recording's timing, and with `--partial` it accepts a score that covers only part of the recording. On four benchmark pieces its alignment puts 90% of notes within 150 ms of the annotated onset (median 49 ms), and its found-rate came out 2 to 7 points below the annotation-based figure on three of them and about 20 below on the fourth, so read its numbers as a lower bound.

## Turning sheet music into a dataset

When a score exists, it is a better reference than any annotation made by ear, so there is a path from page images to a table of notes. `scripts/build_score_dataset.sh` takes a PDF or a folder of page images, reads the pages several at a time with the [oemer](https://github.com/BreezeWhite/oemer) score reader, joins the pages into one score with a named part per staff, and, given a recording, places every note on that recording's clock.

```bash
scripts/build_score_dataset.sh score.pdf my_piece --layout "Violin 1,Violin 2,Piano,Piano" --audio recording.wav
```

Each piece ends up in `datasets/my_piece/` as the page images, the reader's output per page, a `score.mid`, a `notes_score.csv` (part, measure, beat, duration, pitch) and a `notes_aligned.csv` (part, start, end, pitch), the last in the same shape as the URMP annotations. The reader runs in its own environment (`.venv_omr`) because it needs older versions of NumPy and OpenCV than the rest. The reader decides which staves belong together from the barlines that cross them, so a system that is one long measure, with no barline inside it, comes out scrambled; `scripts/helper_barlines.py` finds such systems and draws one extra barline at the emptiest column near the middle, and the two halves are merged back when the pages are joined. On the fifteen-page test score that turned five unreadable pages into fifteen joined ones. A page whose staves still cannot be placed is skipped and listed in `score_meta.json`, never guessed at. Rhythms are the weak part of what it reads, with roughly six bars in ten adding up to a full measure on a clean engraving, and nothing is hand-checked, so treat the result as a draft reference until someone has looked it over.

Key signatures are put right while the pages are joined, by the rules a player reads by and from the notation alone, so that the recording stays free to test the result. The reader keeps one running signature and changes it whenever a measure opens with sharps or flats in every staff it is looking at, which goes wrong in three ways. The courtesy signature printed after the last barline of a line, announcing a change, comes out as one more measure; a last measure that holds a signature and no notes in any staff of the system is dropped. The staves of one system must share a signature, so where the count differs between them the one that continues the previous line wins. And a change inside a line counts only if every staff shows it at the same measure, taking its value from the clean signature at the start of the next line, because the cancelling naturals printed at the change spoil the count there. On the fifteen-page test score, which moves between four flats and four sharps three times, that removed three false measures and respelled 13 notes, and the recording is louder at the new pitch than the old for 10 of the 13 (a median of 6.8 dB, against 4.4 dB for untouched notes compared with a semitone away). A fourth rule did not survive that test: a change that only some staves show was first treated as a misreading and its measure respelled, and the recording preferred the reader's spelling for 15 of those 17 notes. The marks were real accidentals on the first notes of the measure, which hold to the barline, so that measure is now kept as read and only the measures after it return to the signature in force. All of this moves little: key signatures account for about one note in two hundred of this answer key, and the share of its notes that either note source confirms stays at 66%. `--no-key-repair` keeps the reader's signatures, and every decision is listed in `score_meta.json`.

Rhythm is repaired the same way. The reader often misjudges a note's length, and every note after it in that staff's bar is then displaced, so that notes appear beyond the barline (a 4/4 bar read as 4½ or 5 beats). A staff's bar whose notes run past the barline is stretched to fit. A cleverer rule was tried first, choosing the single length correction (a dot added or lost, a value halved or doubled, a triplet) that lines the bar up with the other staves of the same bar where those add up; on the notes it moved, the recording agreed less often than with the plain stretch (44% against 46% found by the instrument-aware model). On Smoke Gets in Your Eyes, the fifteen-page test score, the stretch touched 90 staff-bars, raised the share of bars that add up from 61% to 86%, and raised the instrument-aware model's measured recall and precision there from 60.7% and 63.1% to 65.4% and 68.1%, with the chance level unchanged: the transcription had been better than the old answer key could show. `--no-rhythm-repair` keeps the reader's rhythms, and each stretched bar is listed in `score_meta.json`.

## A second opinion on who is playing

The duet tool also asks a question that does not depend on pitch movement: which violin does this note sound like? It learns each instrument's tone from the passages where both play at once, with no labels, and compares that opinion with the split. On the benchmark the split is right for 90% of notes where the two opinions agree and 71% where they disagree, so disagreements are coloured red in the MusicXML as the places to check by ear first. Using tone to overrule the split, rather than to flag it, did not help: 88.6% against 88.8% without it (`scripts/tone_experiment.py`).

## What is in here

- `scripts/amt.py` holds the shared pieces: pitch tracking and note cutting for single lines, the multi-note detector, beat grid, chords, drums, MIDI and MusicXML export.
- `scripts/pop.py` and `scripts/run_pop.sh` are the pop path.
- `scripts/duet.py` is the two-instrument path, including the part splitter.
- `scripts/run_yourmt3.py` runs the instrument-aware model whose output `duet.py` can take in place of its own detector.
- `scripts/run_beats.py` runs the Beat This! beat and bar tracker.
- `scripts/eval_urmp.py`, `scripts/eval_score.py` and `scripts/eval_ensemble.py` are the checkers: against the benchmark's annotations, against a score read from pages, and against a machine-readable score for any number of players.
- `scripts/eval_detectors.py` gives recall and precision for both note sources on one recording, and for the notes they agree and disagree on.
- `scripts/run_vocal.py` runs two published models for unaccompanied vocal ensembles (Cuesta, McFee and Gómez, 2020; Cuesta and Gómez, 2022; both MIT) from their own repositories under `tools/`, in a separate environment (`.venv_vocal`). On the chorale above it found 80% of the notes with 62% of what it wrote in the score, against 41 to 52% and 40% for the two note sources used elsewhere here. It is a first test on one recording and is not wired into the other tools.
- `scripts/build_score_dataset.sh`, `scripts/helper_barlines.py`, `scripts/omr_stitch.py` and `scripts/align_dataset.py` are the sheet-music path.
- `scripts/tone_experiment.py` is the tone experiment behind the second opinion, `scripts/rearticulation_experiment.py` the one behind the held-note check, and `scripts/instrument_experiment.py` an attempt to tell struck notes from bowed ones that did not work well enough to use (69% against 59% for guessing).
- `scripts/plot_line.py` and `scripts/probe_stereo.py` are small diagnostics.

No audio, stems, scores or transcriptions of commercial recordings are included.
