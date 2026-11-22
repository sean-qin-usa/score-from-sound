"""Check a two-violin transcription against sheet music (MusicXML, .mxl or MIDI).

The score is stretched onto the recording's timing (chroma alignment), then each written
note is looked for in the transcription.

usage: python -I scripts/eval_score.py <score> <audio> <out_dir> [--parts 0,1] [--tol 0.15]
--parts: which staves of the score are the two violins (0-based, default: first two)
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import duet  # noqa: E402
import eval_urmp  # noqa: E402

HOP = 1024


def read_score(path):
    """Return [(part name, [(start, end, midi)])] in nominal seconds (tempo is irrelevant later)."""
    path = Path(path)
    parts = []
    if path.suffix.lower() in (".mid", ".midi"):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            if not inst.is_drum:
                parts.append((inst.name or "part", sorted((n.start, n.end, n.pitch) for n in inst.notes)))
        return parts
    from music21 import converter

    sc = converter.parse(str(path))
    try:
        sc = sc.expandRepeats()
    except Exception:
        pass  # no repeats, or repeats music21 cannot unfold
    for p in sc.parts:
        notes = []
        for el in p.stripTies().flatten().notes:
            if getattr(el.duration, "isGrace", False) or el.quarterLength == 0:
                continue
            for pitch in el.pitches:
                notes.append((float(el.offset), float(el.offset + el.quarterLength), int(pitch.midi)))
        parts.append((p.partName or "part", sorted(notes)))
    return parts


def roll_chroma(notes, n_frames, sr=amt.SR):
    c = np.zeros((12, n_frames))
    for s, e, p in notes:
        a, b = int(s * sr / HOP), max(int(e * sr / HOP), int(s * sr / HOP) + 1)
        c[p % 12, a:b] += 1.0
        c[(p + 7) % 12, a:b] += 0.25  # third harmonic lands a fifth up
    return c


def align(score_notes, y, sr=amt.SR, onset_w=0.0, smooth=0, partial=False):
    """Map score time to recording time. Returns (warp function, mean alignment cost).

    onset_w  weight of a thirteenth feature marking where notes begin (score) and where the
             recording has attacks, so held chords and rests do not let the path drift
    smooth   median-filter the path over this many frames to remove jitter
    partial  the score covers only part of the recording: find where it fits instead of
             stretching it from the first second to the last
    The returned function has a .span attribute: the (start, end) seconds of the recording
    that the score was matched to.
    """
    # The path may bend the tempo by at most a factor of two either way, so the score is first
    # stretched to the length of the music; otherwise a movement played at half the score
    # file's nominal tempo cannot be aligned at all. The music is the stretch between the first
    # and last moments that are audible: a recording usually opens with silence and ends with
    # reverberation, and pinning the score's first note to the file's first second shifted
    # the whole opening (0.8 s on the piece with piano).
    t0, t1 = (0.0, len(y) / sr) if partial else sounding_span(y, sr)
    y = y[int(t0 * sr) : int(t1 * sr)]
    scale = 1.0 if partial else (len(y) / sr) / max(e for _, e, _ in score_notes)
    score_notes = [(s0 * scale, e0 * scale, p0) for s0, e0, p0 in score_notes]
    end = max(e for _, e, _ in score_notes)
    n_s = int(end * sr / HOP) + 1
    S = roll_chroma(score_notes, n_s)
    A = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    S = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-6)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-6)
    if onset_w:
        so = np.zeros(n_s)
        for s0, _, _ in score_notes:
            so[min(int(s0 * sr / HOP), n_s - 1)] = 1.0
        so = np.maximum(so, 0.5 * np.r_[so[1:], 0], 0.5 * np.r_[0, so[:-1]])
        ao = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)[: A.shape[1]]
        ao = np.clip(ao / (np.percentile(ao, 95) + 1e-9), 0, 1)
        S = np.vstack([S, onset_w * so[None, :]])
        A = np.vstack([A, onset_w * np.pad(ao, (0, A.shape[1] - len(ao)))[None, :]])
    D, wp = librosa.sequence.dtw(X=S + 1e-3, Y=A + 1e-3, metric="euclidean", subseq=partial, step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]))
    wp = wp[::-1]
    cost = float((D[wp[-1][0], wp[-1][1]] - (D[wp[0][0], wp[0][1]] if partial else 0.0)) / len(wp))
    st, at = wp[:, 0] * HOP / sr, wp[:, 1] * HOP / sr
    keep = np.r_[True, np.diff(st) > 0]
    st, at = st[keep], at[keep]
    if smooth:
        import scipy.ndimage as ndi

        trend = at[0] + (st - st[0]) * (at[-1] - at[0]) / max(st[-1] - st[0], 1e-9)
        at = np.maximum.accumulate(ndi.median_filter(at - trend, size=smooth, mode="nearest") + trend)

    at = at + t0

    def warp(t):
        return np.interp(np.asarray(t) * scale, st, at)

    warp.span = (float(at[0]), float(at[-1]))
    return warp, cost


def sounding_span(y, sr=amt.SR, floor_db=-40.0):
    """(first, last) second at which the recording is within `floor_db` of its loudest moment."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    loud = np.flatnonzero(20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12) > floor_db)
    if not len(loud):
        return 0.0, len(y) / sr
    return float(loud[0] * HOP / sr), float(min((loud[-1] + 1) * HOP / sr, len(y) / sr))


def _note_roll(notes, n_frames, sr=amt.SR, lo=28, hi=104):
    """Frames x pitches picture of a set of notes: a strong mark where each note starts (so
    timing is sharp), a faint one while it lasts, and a faint copy folded to one octave (so a
    note written an octave off still helps)."""
    R = np.zeros((hi - lo + 12, n_frames), dtype=np.float32)
    for s0, e0, p0 in notes:
        if not lo <= p0 < hi:
            continue
        a = min(int(s0 * sr / HOP), n_frames - 1)
        b = min(max(int(e0 * sr / HOP), a + 1), n_frames)
        R[p0 - lo, a:b] += 0.25
        R[p0 - lo, max(a - 1, 0) : min(a + 2, n_frames)] += 1.0
        R[hi - lo + p0 % 12, a:b] += 0.15
    return R


def align_to_notes(score_notes, heard_notes, length_sec, sr=amt.SR, smooth=5):
    """Map score time to recording time by matching the score's notes to notes heard in the
    recording (from any note detector), instead of matching harmony.

    Harmony changes slowly in slow music, so a harmony-based path can slide within a chord;
    individual note starts pin the timing down. `heard_notes` are (start, end, pitch) in
    seconds and should come from a detector other than the transcription being judged.
    The score is laid over the stretch from the first heard note to the end of the last one,
    not over the whole file, so silence before the music or reverberation after it does not
    shift the opening or the close.
    Returns (warp function with .span, mean cost)."""
    t0 = min((s0 for s0, _, _ in heard_notes), default=0.0)
    t1 = max((e0 for _, e0, _ in heard_notes), default=length_sec)
    length_sec = t1 - t0
    heard_notes = [(s0 - t0, e0 - t0, p0) for s0, e0, p0 in heard_notes]
    scale = length_sec / max(e for _, e, _ in score_notes)
    scaled = [(s0 * scale, e0 * scale, p0) for s0, e0, p0 in score_notes]
    n = int(length_sec * sr / HOP) + 1
    S, A = _note_roll(scaled, n) + 1e-3, _note_roll(heard_notes, n) + 1e-3
    S /= np.linalg.norm(S, axis=0, keepdims=True)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    D, wp = librosa.sequence.dtw(X=S, Y=A, metric="euclidean", step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]))
    wp = wp[::-1]
    cost = float(D[wp[-1][0], wp[-1][1]] / len(wp))
    st, at = wp[:, 0] * HOP / sr, wp[:, 1] * HOP / sr
    keep = np.r_[True, np.diff(st) > 0]
    st, at = st[keep], at[keep]
    if smooth:
        import scipy.ndimage as ndi

        trend = at[0] + (st - st[0]) * (at[-1] - at[0]) / max(st[-1] - st[0], 1e-9)
        at = np.maximum.accumulate(ndi.median_filter(at - trend, size=smooth, mode="nearest") + trend)

    at = at + t0

    def warp(t):
        return np.interp(np.asarray(t) * scale, st, at)

    warp.span = (float(at[0]), float(at[-1]))
    return warp, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--parts", default="0,1")
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--min-ms", type=float, default=58.0)
    ap.add_argument("--partial", action="store_true", help="the score covers only part of the recording")
    ap.add_argument("--check-overtones", action="store_true", help="remove notes the recording says are only an overtone of a lower note")
    ap.add_argument("--no-stereo", action="store_true", help="ignore left/right position even when the recording is stereo")
    ap.add_argument("--keep-rewrites", action="store_true", help="do not join a held note that the detector wrote twice")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = read_score(args.score)
    print("score staves:", [(i, n, len(notes)) for i, (n, notes) in enumerate(parts)])
    vi = [int(x) for x in args.parts.split(",")]
    y = amt.load_mono(args.audio)
    warp, cost = align([n for _, notes in parts for n in notes], y, onset_w=1.0, smooth=9, partial=args.partial)
    lo, hi = warp.span

    truth, others = [], []
    for i, (_, notes) in enumerate(parts):
        for s, e, p in notes:
            row = dict(start=float(warp(s)), end=float(warp(e)), pitch=p, player=vi.index(i) if i in vi else -1)
            (truth if i in vi else others).append(row)
    truth.sort(key=lambda g: g["start"])

    raw = amt.poly_notes(args.audio, fmin=185.0, fmax=4200.0, min_ms=args.min_ms)
    if not args.keep_rewrites:
        raw = amt.merge_held(raw, y)
    if args.check_overtones:
        raw = amt.drop_checked_overtones(raw, y)
    A, B, how = duet.split_parts(list(raw), args.audio, use_stereo=not args.no_stereo)
    A, B = duet.trim_tails(A), duet.trim_tails(B)
    A = [n for n in A if lo - args.tol <= n.start <= hi + args.tol]  # judge only where the score reaches
    B = [n for n in B if lo - args.tol <= n.start <= hi + args.tol]
    final = A + B
    label = {id(n): 0 for n in A} | {id(n): 1 for n in B}
    eval_urmp.TOL = args.tol
    m = eval_urmp.match(truth, final)
    found, used = {r for r, _ in m}, {e for _, e in m}
    agree = sum(1 for r, e in m if truth[r]["player"] == label[id(final[e])])
    extra = [n for j, n in enumerate(final) if j not in used]
    from_other = sum(1 for n in extra if any(abs(o["start"] - n.start) <= args.tol and o["pitch"] == n.pitch for o in others))
    high_blips = sum(1 for n in extra if n.dur < 0.15 and n.pitch >= 84)
    in_range = [g for g in truth if g["pitch"] >= 55]
    report = {
        "how_parts_were_split": how,
        "alignment_cost": cost,
        "recording_seconds_covered_by_score": [round(lo, 1), round(hi, 1)],
        "recording_length_sec": round(len(y) / amt.SR, 1),
        "violin_notes_in_score": len(truth),
        "found": len(m) / max(len(truth), 1),
        "written_notes_that_are_in_violin_parts": len(m) / max(len(final), 1),
        "right_player": max(agree, len(m) - agree) / max(len(m), 1),
        "extra_notes": len(extra),
        "extra_notes_that_are_another_staff": from_other,
        "extra_short_notes_above_C6": high_blips,
    }
    for k, name in enumerate(parts[i][0] for i in vi):
        idx = [i for i, g in enumerate(truth) if g["player"] == k]
        report["found_in_%s_(staff %d)" % (name, vi[k])] = sum(i in found for i in idx) / max(len(idx), 1)
    print(json.dumps(report, indent=1))
    (out / "score_check.json").write_text(json.dumps(report, indent=1))
    return truth, warp


if __name__ == "__main__":
    main()
