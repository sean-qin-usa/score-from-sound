"""Shared transcription helpers: note extraction, beat grid, MIDI/MusicXML export."""
from dataclasses import dataclass

import librosa
import math

import numpy as np
import pretty_midi
import scipy.ndimage as ndi
import scipy.signal as sps

SR = 22050
HOP = 256
PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    amp: float = 0.8
    conf: float = 1.0  # 0..1, how sure the pitch is (single-line parts only)
    off: float = 0.0  # how far the sung pitch sat from the written note, in semitones (+ = sharp)
    eff: float = 0.0  # length of the held note this fragment belongs to (see duet.effective_durations)
    seen: float = 1.0  # share of the independent note sources that wrote this note (see duet.cross_check)

    @property
    def dur(self):
        return self.end - self.start


def name(p):
    return "%s%d" % (PITCH_NAMES[p % 12], p // 12 - 1)


def load_mono(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


# ---------------------------------------------------------------- single-line parts
def track_f0(y, fmin, fmax, sr=SR, gate_db=-40.0, cache=None):
    """pYIN pitch curve in (fractional) MIDI numbers, NaN where nothing is sung; plus loudness and
    the tracker's own per-frame probability that a pitched sound is present."""
    if cache is not None and cache.exists():
        d = np.load(cache)
        if "prob" in d:
            return d["midi"], d["rms"], d["prob"]
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=HOP, fill_na=np.nan)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][: len(f0)]
    loud = 20 * np.log10(rms / (rms.max() + 1e-9) + 1e-9) > gate_db
    ok = voiced & loud & np.isfinite(f0)
    midi = np.full(len(f0), np.nan)
    midi[ok] = librosa.hz_to_midi(f0[ok])
    if cache is not None:
        np.savez(cache, midi=midi, rms=rms, prob=prob)
    return midi, rms, prob


def segment_notes(midi, rms, prob=None, sr=SR, min_dur=0.12, smooth_frames=11, leave=0.65, hold_frames=5, max_gap=3):
    """Cut a pitch curve into notes.

    A note continues until the smoothed pitch has left its semitone by more than `leave`
    for `hold_frames` in a row, so vibrato and scoops do not split it.
    Returns (notes, tuning_offset_in_semitones).
    """
    ok = np.isfinite(midi)
    if ok.sum() < 10:
        return [], 0.0
    dev = midi[ok] - np.round(midi[ok])
    hist, edges = np.histogram(dev, bins=20, range=(-0.5, 0.5))
    tuning = float(edges[np.argmax(hist)] + 0.025)  # recording may sit between piano keys
    midi = midi - tuning
    idx = np.where(ok)[0]
    runs = np.split(idx, np.where(np.diff(idx) > max_gap)[0] + 1)
    min_frames = max(2, int(min_dur * sr / HOP))
    t = lambda i: i * HOP / sr
    notes = []
    for r in runs:
        a, b = r[0], r[-1] + 1
        seg = midi[a:b].copy()
        bad = ~np.isfinite(seg)
        if bad.any():
            seg[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], seg[~bad])
        s = ndi.median_filter(seg, size=smooth_frames, mode="nearest")
        segs, start, center, away = [], 0, int(round(s[0])), 0
        for i in range(len(s)):
            away = away + 1 if abs(s[i] - center) > leave else 0
            if away >= hold_frames:
                cut = i - hold_frames + 1
                if cut > start:
                    segs.append([start, cut, center])
                start, center, away = cut, int(round(np.median(s[cut : i + 1]))), 0
        segs.append([start, len(s), center])
        # fold slides and blips into the neighbour that is closer in pitch
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for i, (s0, e0, p) in enumerate(segs):
                if e0 - s0 < min_frames:
                    left = segs[i - 1] if i > 0 else None
                    right = segs[i + 1] if i + 1 < len(segs) else None
                    if left and (not right or abs(left[2] - p) <= abs(right[2] - p)):
                        left[1] = e0
                    else:
                        right[0] = s0
                    segs.pop(i)
                    changed = True
                    break
        merged = []
        for s0, e0, p in segs:
            if merged and merged[-1][2] == p:
                merged[-1][1] = e0
            else:
                merged.append([s0, e0, p])
        prev_end = None
        for s0, e0, p in merged:
            if e0 - s0 < min_frames:
                continue
            med = np.median(s[s0:e0])
            held = np.mean(np.abs(s[s0:e0] - med) <= 0.5)
            if held < 0.6:
                # the pitch never settles: a slide between notes or a fall at the end of a phrase.
                # Let the previous note ring through it instead of inventing stair-step notes.
                if prev_end == s0 and notes:
                    notes[-1].end = t(a + e0)
                    prev_end = e0
                continue
            amp = float(np.clip(rms[a + s0 : a + e0].mean() / (rms.max() + 1e-9) * 2, 0.3, 1.0))
            # confidence: the tracker is sure there is a pitch, and the pitch sits still on one note
            sure = float(np.mean(prob[a + s0 : a + e0])) if prob is not None else 1.0
            notes.append(Note(t(a + s0), t(a + e0), int(round(med)), amp, conf=sure * float(held), off=float(med - round(med))))
            prev_end = e0
    # an isolated note an octave away from both neighbours is a tracker slip, not a leap
    for i in range(1, len(notes) - 1):
        lo, n, hi = notes[i - 1], notes[i], notes[i + 1]
        for shift in (12, -12):
            if abs(n.pitch + shift - lo.pitch) <= 4 and abs(n.pitch + shift - hi.pitch) <= 4 and abs(n.pitch - lo.pitch) >= 8 and n.dur < 0.6:
                n.pitch += shift
                break
    return notes, tuning


def mono_notes(y, fmin, fmax, cache=None, **kw):
    midi, rms, prob = track_f0(y, fmin, fmax, cache=cache)
    return segment_notes(midi, rms, prob, **kw)


_DIATONIC = (0, 2, 4, 5, 7, 9, 11)


def best_scale(weights):
    """The seven-note major-scale set (any of 12 transpositions) that holds the most weight."""
    w = np.asarray(weights, float)
    w = w / (w.sum() + 1e-9)
    fits = [(sum(w[(r + k) % 12] for k in _DIATONIC), r) for r in range(12)]
    fit, root = max(fits)
    return {(root + k) % 12 for k in _DIATONIC}, PITCH_NAMES[root] + " major scale notes", float(fit), float(sorted(fits)[-2][0])


def autotune(notes, scale):
    """Move notes that fall outside the scale by one semitone onto it.

    The note goes to the side the singer was leaning toward when that side is in the scale,
    otherwise to whichever neighbour is. Notes already in the scale are left alone."""
    moved = 0
    for n in notes:
        if n.pitch % 12 in scale:
            continue
        lean = 1 if n.off >= 0 else -1
        for d in (lean, -lean):
            if (n.pitch + d) % 12 in scale:
                n.pitch += d
                moved += 1
                break
    merged = []
    for n in notes:  # two notes that became the same pitch back to back are one note
        if merged and merged[-1].pitch == n.pitch and n.start - merged[-1].end <= 0.03:
            merged[-1].end = n.end
            merged[-1].conf = max(merged[-1].conf, n.conf)
        else:
            merged.append(n)
    return merged, moved


def settle_uncertain(notes, threshold, max_step=2, gap=0.06):
    """Clean up notes the tracker was unsure about (spoken or breathy singing).

    1. An unsure note that follows a sure one within `max_step` semitones is treated as that
       note still sounding (sustain) instead of a new pitch.
    2. Any other unsure note is pulled onto the nearest pitch the singer clearly uses elsewhere.
    Unsure notes keep conf < threshold so they can be shown differently in the score.
    """
    sure = [n for n in notes if n.conf >= threshold]
    w = np.zeros(12)
    for n in sure:
        w[n.pitch % 12] += n.dur * n.conf
    allowed = set(np.argsort(w)[::-1][:7]) if w.sum() > 0 else set(range(12))
    out, sustained, moved = [], 0, 0
    for n in notes:
        if n.conf >= threshold:
            out.append(n)
            continue
        prev = out[-1] if out else None
        if prev is not None and prev.conf >= threshold and n.start - prev.end <= gap and abs(n.pitch - prev.pitch) <= max_step:
            prev.end = n.end
            sustained += 1
            continue
        if n.pitch % 12 not in allowed:
            for d in (1, -1, 2, -2):
                if (n.pitch + d) % 12 in allowed:
                    n.pitch += d
                    moved += 1
                    break
        out.append(n)
    return out, {"sustained_into_previous": sustained, "moved_to_scale": moved, "scale_pitch_classes": sorted(PITCH_NAMES[i] for i in allowed)}


def synth(notes, length, sr=SR, shift=0, partials=(1, 2, 3)):
    """Plain tone rendering of notes, for side-by-side listening against the recording."""
    out = np.zeros(length, dtype=np.float32)
    for n in notes:
        a, b = int(n.start * sr), min(int(n.end * sr), length)
        if b - a < 32:
            continue
        tt = np.arange(b - a) / sr
        f = 440.0 * 2 ** ((n.pitch + shift - 69) / 12)
        w = sum(np.sin(2 * np.pi * f * k * tt) / k for k in partials) * 0.2 * n.amp
        fade = min(256, (b - a) // 2)
        w[:fade] *= np.linspace(0, 1, fade)
        w[-fade:] *= np.linspace(1, 0, fade)
        out[a:b] += w.astype(np.float32)
    return out


def write_check(path, original, notes, sr=SR, shift=0):
    """Stereo file: recording in the left ear, transcription in the right.

    `shift` plays the transcription that many semitones higher (a bass line an octave up is
    audible on small speakers; the written notes are unchanged)."""
    import soundfile as sf

    tone = synth(notes, len(original), sr, shift=shift)
    o = original / (np.abs(original).max() + 1e-9) * 0.7
    tone = tone / (np.abs(tone).max() + 1e-9) * 0.6
    sf.write(str(path), np.stack([o, tone], axis=1), sr)


# ---------------------------------------------------------------- multi-note parts
_BP_MODEL = None


def poly_notes(path, fmin=None, fmax=None, onset=0.5, frame=0.3, min_ms=110, gate_db=-40.0):
    """Transcribe several simultaneous notes with Basic Pitch (ONNX, CPU)."""
    global _BP_MODEL
    import basic_pitch
    from basic_pitch.inference import Model, predict

    if _BP_MODEL is None:
        _BP_MODEL = Model(basic_pitch.build_icassp_2022_model_path(basic_pitch.FilenameSuffix.onnx))
    _, _, events = predict(
        str(path),
        _BP_MODEL,
        onset_threshold=onset,
        frame_threshold=frame,
        minimum_note_length=min_ms,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        melodia_trick=True,
    )
    notes = [Note(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]
    if gate_db is not None and notes:
        # the model also "hears" notes in near-silent leftovers of a stem; keep only notes
        # where the recording itself is audible
        y = load_mono(path)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        db = 20 * np.log10(rms / (rms.max() + 1e-9) + 1e-9)
        f = lambda t: int(t * SR / 512)
        notes = [n for n in notes if db[f(n.start) : max(f(n.end), f(n.start) + 1)].mean() > gate_db]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


# ---------------------------------------------------------------- held notes written twice
def level_track(S, freqs, pitch):
    """Level in dB over time of one pitch: its first six harmonics added together."""
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    e = np.zeros(S.shape[1])
    for k in range(1, 7):
        if f0 * k > 9000:
            break
        m = (freqs > f0 * k * 0.97) & (freqs < f0 * k * 1.03)
        if m.sum():
            e += (S[m] ** 2).sum(axis=0)
    return 10 * np.log10(e + 1e-10)


def attack_evidence(track, t, hop, sr=SR):
    """(rise, dip) of a pitch's level around time t: the biggest jump over two frames, and how
    far the level falls below what surrounds it. A new bow stroke or key strike shows both."""
    c = int(t * sr / hop)
    a, b = max(c - 6, 1), min(c + 8, len(track) - 1)
    if b - a < 4:
        return None
    w = track[a:b]
    before = track[max(c - 14, 0) : max(c - 5, 1)]
    after = track[min(c + 6, len(track) - 1) : min(c + 16, len(track))]
    ref = min(before.mean() if len(before) else w.mean(), after.mean() if len(after) else w.mean())
    return float(np.max(w[2:] - w[:-2])), float(ref - w.min())


def merge_held(notes, y, rise_min=4.2, dip_min=4.0, gap=0.08, sr=SR):
    """Join a note onto the previous note of the same pitch unless the recording shows a new
    attack there. The detector often writes one held note as two or three; a blind merge
    would also swallow real repeated notes, so the recording decides.

    Thresholds were chosen on half of the URMP two-violin pieces (a real attack has a median
    rise of 6.8 dB and dip of 6.3 dB; a continuation 1.9 and -0.1) and checked on the rest."""
    hop = 256
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tracks, out, last = {}, [], {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        p = last.get(n.pitch)
        if p is not None and n.start - p.end <= gap:
            if n.pitch not in tracks:
                tracks[n.pitch] = level_track(S, freqs, n.pitch)
            f = attack_evidence(tracks[n.pitch], n.start, hop)
            if f is not None and f[0] < rise_min and f[1] < dip_min:
                p.end = max(p.end, n.end)
                p.amp = max(p.amp, n.amp)
                continue
        out.append(n)
        last[n.pitch] = n
    return out


# ---------------------------------------------------------------- overtones written as notes
OVERTONE_STEPS = {12: 2, 19: 3, 24: 4, 28: 5}  # semitones above -> which partial of the lower note


def overtone_spectrogram(y, sr=SR):
    hop = 256
    return np.abs(librosa.stft(y, n_fft=4096, hop_length=hop)), librosa.fft_frequencies(sr=sr, n_fft=4096), hop


def overtone_candidates(notes):
    """(note, lower note, partial number) for each note that sits on an overtone of another
    note sounding for at least half of its length. The loudest such lower note is used."""
    out = []
    by_start = sorted(notes, key=lambda n: n.start)
    for n in by_start:
        best = None
        for m in by_start:
            if m.start > n.end:
                break
            ratio = OVERTONE_STEPS.get(n.pitch - m.pitch)
            if ratio and min(m.end, n.end) - max(m.start, n.start) > 0.5 * n.dur:
                if best is None or m.amp > best[0].amp:
                    best = (m, ratio)
        if best:
            out.append((n, best[0], best[1]))
    return out


def _partial_db(S, freqs, f, a, b):
    m = (freqs > f * 0.97) & (freqs < f * 1.03)
    if not m.sum() or b <= a:
        return None
    return 10 * np.log10((S[m, a:b] ** 2).sum(axis=0).mean() + 1e-10)


def overtone_features(S, freqs, hop, n, parent, ratio, sr=SR):
    """[stickout, amp ratio, onset lateness, length ratio, own rise] for one candidate."""
    f0 = 440.0 * 2 ** ((parent.pitch - 69) / 12)
    a, b = int(max(n.start, parent.start) * sr / hop), int(min(n.end, parent.end) * sr / hop)
    if b - a < 3:
        return None
    stick = []
    for k in range(ratio, 13, ratio):  # the lower note's partials that the candidate shares
        if f0 * (k + 1) > 9000:
            break
        here = _partial_db(S, freqs, f0 * k, a, b)
        side = [_partial_db(S, freqs, f0 * j, a, b) for j in (k - 1, k + 1) if j >= 1]
        side = [x for x in side if x is not None]
        if here is not None and side:
            stick.append(here - float(np.mean(side)))
    if not stick:
        return None
    c = int(n.start * sr / hop)
    lo, hi = max(c - 6, 0), min(c + 8, S.shape[1] - 1)
    if hi - lo < 4:
        return None

    def rise(ks):
        e = np.zeros(hi - lo)
        for k in ks:
            m = (freqs > f0 * k * 0.97) & (freqs < f0 * k * 1.03)
            if m.sum() and f0 * k < 9000:
                e += (S[m, lo:hi] ** 2).sum(axis=0)
        t = 10 * np.log10(e + 1e-10)
        return float(np.max(t[2:] - t[:-2]))

    shared = [k for k in range(ratio, 13, ratio)]
    others = [k for k in range(1, 13) if k % ratio]
    return [float(np.mean(stick)), n.amp / max(parent.amp, 1e-6), n.start - parent.start, n.dur / max(parent.dur, 1e-6), rise(shared) - rise(others)]


# Fitted on 356 candidates from the URMP two-violin pieces (scripts/overtone_experiment.py):
# a logistic model on the five features above, with the cut-off set to keep 95% of real notes.
_OT_MEAN = np.array([8.4434, 0.7354, 0.1642, 0.6069, 1.2447])
_OT_SCALE = np.array([5.1661, 0.3004, 0.3334, 0.4061, 2.709])
_OT_COEF = np.array([1.9981, 1.3242, 0.4875, 0.6039, -0.3447])
_OT_INTERCEPT, _OT_CUT = 0.9770, 0.195


def drop_checked_overtones(notes, y):
    """Remove notes the recording says are only an overtone of a lower note.

    Held out on the benchmark this keeps about 91% of real upper notes and catches 65-77% of
    overtones, which on two violins costs about five points on the upper note of an octave for
    two points of accuracy: not worth it there. With a piano in the mix it removed 143 notes
    on the test piece, 112 of which matched nothing in the score and 11 of which were violin
    notes, so it is offered as an option for accompanied pieces."""
    S, freqs, hop = overtone_spectrogram(y)
    drop = set()
    for n, parent, ratio in overtone_candidates(notes):
        f = overtone_features(S, freqs, hop, n, parent, ratio)
        if f is None:
            continue
        z = float(((np.array(f) - _OT_MEAN) / _OT_SCALE) @ _OT_COEF + _OT_INTERCEPT)
        if 1.0 / (1.0 + np.exp(-z)) < _OT_CUT:
            drop.add(id(n))
    return [n for n in notes if id(n) not in drop]


# ---------------------------------------------------------------- left/right position
def pan_per_note(path, notes, sr=SR):
    """Right-minus-left level in dB at each note's first four harmonics, or None when the
    recording is mono (or its two channels are the same signal)."""
    import soundfile as sf

    st, file_sr = sf.read(str(path), always_2d=True)
    if st.shape[1] < 2 or np.corrcoef(st[:, 0], st[:, 1])[0, 1] > 0.98:
        return None
    hop = 512
    chans = [np.abs(librosa.stft(librosa.resample(st[:, c].astype(np.float32), orig_sr=file_sr, target_sr=sr), n_fft=4096, hop_length=hop)) ** 2 for c in (0, 1)]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    out = []
    for n in notes:
        a = int(n.start * sr / hop)
        b = max(int(n.end * sr / hop), a + 2)
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        e = [0.0, 0.0]
        for k in (1, 2, 3, 4):
            m = (freqs > f0 * k * 0.97) & (freqs < f0 * k * 1.03)
            for c in (0, 1):
                e[c] += float(chans[c][m, a:b].sum())
        out.append(10 * np.log10((e[1] + 1e-12) / (e[0] + 1e-12)))
    return out


# ---------------------------------------------------------------- rhythm grid
def beat_grid(y, sr=SR, start_bpm=90):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, start_bpm=start_bpm, units="time")
    beats = np.asarray(beats, dtype=float)
    ibi = np.diff(beats)
    bpm = 60.0 / np.median(ibi)
    jitter = float(np.std(ibi) / np.mean(ibi))
    return beats, float(bpm), jitter


def to_beats(t, beats):
    """Map seconds to (fractional) beat index, extrapolating at both ends."""
    step = np.median(np.diff(beats))
    ext = np.r_[beats[0] - 64 * step, beats, beats[-1] + 64 * step]
    pos = np.r_[-64.0, np.arange(len(beats)), len(beats) - 1 + 64.0]
    return np.interp(t, ext, pos)


def warp(notes, beats, bpm, grid=None):
    """Put notes on a constant-tempo timeline so a notation program reads them on the beat."""
    out = []
    for n in notes:
        s, e = to_beats(n.start, beats), to_beats(n.end, beats)
        if grid:
            s = round(s * grid) / grid
            e = max(round(e * grid) / grid, s + 1.0 / grid)
        out.append(Note(s * 60 / bpm, e * 60 / bpm, n.pitch, n.amp, n.conf, n.off, seen=n.seen))
    return out


def drum_hits(y, sr=SR):
    """Kick / snare / hi-hat hits from a drum stem, as General-MIDI drum notes (36 / 38 / 42).

    Each onset is labelled by where its energy sits: low band = kick, high band = hi-hat,
    otherwise snare. A hit with both strong lows and strong highs is written as kick + hat."""
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, backtrack=False, delta=0.12, wait=4)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    lo, mid, hi = (f >= 30) & (f < 130), (f >= 150) & (f < 3000), f >= 5500
    peak = env.max() + 1e-9
    hits = []
    for fr in frames:
        w = S[:, fr : fr + 6].sum(axis=1)
        tot = w.sum() + 1e-12
        l, m, h = w[lo].sum() / tot, w[mid].sum() / tot, w[hi].sum() / tot
        t, amp = fr * hop / sr, float(np.clip(env[fr] / peak * 1.5, 0.3, 1.0))
        if l > 0.4:
            hits.append(Note(t, t + 0.1, 36, amp))
            if h > 0.2:
                hits.append(Note(t, t + 0.1, 42, amp * 0.7))
        elif h > 0.45 and m < 0.45:
            hits.append(Note(t, t + 0.1, 42, amp * 0.7))
        else:
            hits.append(Note(t, t + 0.1, 38, amp))
    return hits


# ---------------------------------------------------------------- harmony
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_weights(notes):
    w = np.zeros(12)
    for n in notes:
        w[n.pitch % 12] += n.dur
    return w


def estimate_key(chroma_mean):
    best = (-2, "")
    for i in range(12):
        for prof, label in ((_MAJ, "major"), (_MIN, "minor")):
            c = np.corrcoef(np.roll(prof, i), chroma_mean)[0, 1]
            if c > best[0]:
                best = (c, "%s %s" % (PITCH_NAMES[i], label))
    return best[1], float(best[0])


DIATONIC = (0, 2, 4, 5, 7, 9, 11)  # the major scale; a key signature is this set moved round the circle of fifths


def signature_pitches(fifths):
    """Pitch classes a key signature of `fifths` sharps (negative: flats) leaves unaltered."""
    tonic = (7 * fifths) % 12
    return {(tonic + d) % 12 for d in DIATONIC}


def key_by_section(notes, bar_starts, change_cost=2.0, choices=range(-5, 7)):
    """One key signature per bar, the way an editor would choose them.

    A bar's notes (weighted by length) are scored against each signature by the share of
    their length that falls outside it; a signature changes only where the change saves more
    than `change_cost` bars' worth of out-of-key notes, so a chromatic passing note or a short
    borrowed chord never flips the signature, while a section that stays in a new key does.
    The minor mode's raised seventh costs a little everywhere and moves nothing. Bars with no
    notes take whatever is cheapest around them. Returns a list of `fifths`, one per bar."""
    choices = list(choices)
    sets = [signature_pitches(f) for f in choices]
    n = len(bar_starts)
    ends = list(bar_starts[1:]) + [float("inf")]
    cost = np.zeros((n, len(choices)))
    for b, (lo, hi) in enumerate(zip(bar_starts, ends)):
        w = np.zeros(12)
        for x in notes:
            overlap = min(x.end, hi) - max(x.start, lo)
            if overlap > 0:
                w[x.pitch % 12] += overlap
        if w.sum() > 0:
            w /= w.sum()
            cost[b] = [sum(w[pc] for pc in range(12) if pc not in s) for s in sets]
    # fewest accidentals breaks ties, so an empty or ambiguous stretch reads as the plainer key
    cost += 1e-3 * np.abs(np.array(choices))[None, :]
    total, back = cost[0].copy(), np.zeros((n, len(choices)), int)
    for b in range(1, n):
        stay = total
        move = total.min() + change_cost
        back[b] = np.where(stay <= move, np.arange(len(choices)), total.argmin())
        total = np.minimum(stay, move) + cost[b]
    path = [int(total.argmin())]
    for b in range(n - 1, 0, -1):
        path.append(int(back[b][path[-1]]))
    return [choices[i] for i in reversed(path)]


def spell(pitch, fifths):
    """music21 pitch name for a MIDI note in a key signature: the scale's own letter when the
    note is in the key, otherwise a sharp in sharp keys and a flat in flat keys."""
    names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    names_flat = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]
    pc, octave = pitch % 12, pitch // 12 - 1
    if pc in signature_pitches(fifths):
        # in the key: sharp keys spell with sharps, flat keys with flats (and the few that need
        # B#, E#, Cb or Fb do so)
        letters = "CDEFGAB"
        tonic = (7 * fifths) % 12
        steps = [(tonic + d) % 12 for d in DIATONIC]
        start ="CGDAEBF"[fifths % 7] if fifths >= 0 else "CFBEADG"[(-fifths) % 7]
        i = steps.index(pc)
        letter = letters[(letters.index(start) + i) % 7]
        natural = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
        alter = (pc - natural + 6) % 12 - 6
        if letter == "B" and alter == 1:
            octave -= 1
        if letter == "C" and alter == -1:
            octave += 1
        return letter + ("#" * alter if alter > 0 else "-" * -alter) + str(octave)
    return (names_sharp if fifths >= 0 else names_flat)[pc] + str(octave)


def beat_chords(y, beats, sr=SR, per=2):
    """One chord label per `per` beats, from triad templates on beat-averaged chroma."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=36)
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=512)
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)[:, 1:]
    n = sync.shape[1] // per
    grouped = sync[:, : n * per].reshape(12, n, per).mean(axis=2)
    templates, labels = [], []
    for r in range(12):
        for iv, suffix in (((0, 4, 7), ""), ((0, 3, 7), "m")):
            v = np.zeros(12)
            v[[(r + k) % 12 for k in iv]] = 1
            templates.append(v / np.linalg.norm(v))
            labels.append(PITCH_NAMES[r] + suffix)
    T = np.array(templates)
    g = grouped / (np.linalg.norm(grouped, axis=0, keepdims=True) + 1e-9)
    score = T @ g
    return [labels[i] for i in score.argmax(axis=0)], chroma.mean(axis=1)


def pad_from_harmony(y, beats, sr=SR, per=2, voices=3, low=55, floor=0.45):
    """Sustained backing voices taken from what the instruments are playing.

    Every `per` beats the strongest pitch classes in the instrumental are picked (up to `voices`,
    each at least `floor` of the strongest), placed within one octave above `low`, and handed
    to the voices from bottom to top. Repeated pitches are tied into held notes. This is an
    arranging aid: the notes come from the recording's harmony, not from anyone singing them.
    Returns one list of notes per voice, lowest voice first."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=36)
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=512)
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)[:, 1:]
    n = min(sync.shape[1], len(beats) - 1) // per
    out = [[] for _ in range(voices)]
    for k in range(n):
        g = sync[:, k * per : (k + 1) * per].mean(axis=1)
        order = [int(i) for i in np.argsort(g)[::-1] if g[i] >= floor * g.max()][:voices]
        if len(order) < 2:
            continue
        pitches = sorted(low + (pc - low) % 12 for pc in order)
        t0, t1 = float(beats[k * per]), float(beats[(k + 1) * per])
        for v, p in enumerate(pitches):
            last = out[v][-1] if out[v] else None
            if last is not None and last.pitch == p and abs(last.end - t0) < 0.05:
                last.end = t1
            else:
                out[v].append(Note(t0, t1, int(p), 0.6))
    return out


# ---------------------------------------------------------------- checks and export
def chroma_agreement(notes, y, sr=SR):
    """Correlation between audio chroma and the chroma implied by the notes (1.0 = identical)."""
    hop = 2048
    ca = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    roll = np.zeros_like(ca)
    for n in notes:
        a, b = int(n.start * sr / hop), int(np.ceil(n.end * sr / hop))
        roll[n.pitch % 12, a : max(b, a + 1)] += 1
    active = roll.sum(axis=0) > 0
    if active.sum() < 5:
        return float("nan")
    return float(np.corrcoef(ca[:, active].ravel(), roll[:, active].ravel())[0, 1])


def polyphony_profile(notes, step=0.02):
    if not notes:
        return {}
    T = np.arange(0, max(n.end for n in notes), step)
    count = np.zeros(len(T), int)
    for n in notes:
        count[int(n.start / step) : int(n.end / step)] += 1
    sounding = count > 0
    return {k: float((count[sounding] == k).mean()) for k in range(1, 5)} | {
        "5+": float((count[sounding] >= 5).mean())
    }


def write_midi(path, tracks, bpm):
    """tracks: list of (name, General-MIDI program, notes)."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for label, program, notes in tracks:
        inst = pretty_midi.Instrument(program=0, is_drum=True, name=label) if program == "drums" else pretty_midi.Instrument(program=program, name=label)
        for n in notes:
            inst.notes.append(
                pretty_midi.Note(velocity=int(40 + 70 * min(n.amp, 1.0)), pitch=n.pitch, start=max(n.start, 0), end=max(n.end, max(n.start, 0) + 0.05))
            )
        pm.instruments.append(inst)
    pm.write(str(path))


def write_musicxml(path, tracks, bpm, key=None, meter="4/4", unsure_below=None, unsure_style="x"):
    """tracks: list of (name, clef, notes) with note times already on the constant-tempo grid."""
    from music21 import chord as m21chord
    from music21 import clef, instrument, metadata, meter as m21meter, note as m21note, stream, tempo
    from music21 import key as m21key

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    spb = 60.0 / bpm
    # notes before the first beat found would sit before the first bar: move everything later by
    # whole bars instead, so the bar lines stay where they were
    bar = m21meter.TimeSignature(meter).barDuration.quarterLength
    earliest = min((n.start / spb for _, _, notes in tracks for n in notes), default=0.0)
    lead = bar * math.ceil(-earliest / bar) if earliest < 0 else 0.0
    for label, clef_name, notes in tracks:
        part = stream.Part()
        inst = instrument.Instrument()
        inst.partName = label
        part.insert(0, inst)
        part.insert(0, {"treble": clef.TrebleClef(), "treble8vb": clef.Treble8vbClef(), "bass": clef.BassClef()}[clef_name])
        part.insert(0, m21meter.TimeSignature(meter))
        part.insert(0, tempo.MetronomeMark(number=round(bpm)))
        if key:
            tonic, mode = key.split()
            part.insert(0, m21key.Key(tonic.replace("b", "-"), mode))
        by_onset = {}
        for n in notes:
            by_onset.setdefault(round((n.start / spb + lead) * 4) / 4, []).append(n)
        for on, group in sorted(by_onset.items()):
            ql = max(0.25, round(max(g.end - g.start for g in group) / spb * 4) / 4)
            el = m21note.Note(group[0].pitch) if len(group) == 1 else m21chord.Chord(sorted({g.pitch for g in group}))
            el.quarterLength = ql
            if unsure_style == "color" and any(g.seen < 1.0 for g in group):
                el.style.color = "#1f6fd0"  # only one of two note sources wrote this note: it may not be real
            elif unsure_below is not None and unsure_style == "color" and any(g.conf < unsure_below for g in group):
                el.style.color = "#d00000"  # the independent checks disagree about who plays this note
            elif unsure_below is not None and len(group) == 1 and group[0].conf < unsure_below:
                el.notehead = "x"  # pitch is a guess: spoken or unclear
            part.insert(on, el)
        part.makeVoices(inPlace=True, fillGaps=True) if part.hasVoices() else None
        score.insert(0, part.makeMeasures().makeTies().makeRests(fillGaps=True, inPlace=False))
    score.write("musicxml", fp=str(path))
