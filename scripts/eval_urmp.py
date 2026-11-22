"""Score the two-violin transcription against URMP's hand-checked note annotations.

usage: python -I scripts/eval_urmp.py <urmp_dir> <out_dir> [variant ...]
Each piece folder needs AuSep_{1,2}_vn_*.wav and Notes_{1,2}_vn_*.txt (onset sec, pitch Hz, duration sec).
"""
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import duet  # noqa: E402

TOL = 0.10  # seconds: a note counts as found if it starts within this of the annotation

# Each variant changes one thing relative to "current", to test one explanation at a time.
VARIANTS = {
    "current": dict(min_ms=110, onset=0.5, frame=0.3, fmax=3600.0, ghosts=True, trim=True),
    "no_ghost_filter": dict(min_ms=110, onset=0.5, frame=0.3, fmax=3600.0, ghosts=False, trim=True),
    "short_notes": dict(min_ms=58, onset=0.5, frame=0.3, fmax=3600.0, ghosts=True, trim=True),
    "new_default": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True),
    # who-is-who experiments, each adding one idea to new_default
    "stay_in_part": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, split=dict(idle=6.0)),
    "note_length": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, split=dict(dur_w=1.5)),
    "ring_ok": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, split=dict(ring_ok=True)),
    "blip_filter": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, blips=True),
    "leap_filter": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, leap=True),
    "merge_held": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, leap=True, held=True),
    "register_4": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, leap=True, held=True, split=dict(register_w=4.0)),
    "register_8": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, leap=True, held=True, split=dict(register_w=8.0)),
    "heal_lines": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, leap=True, held=True, heal=True, split=dict(register_w=4.0)),
    "merge_repeats": dict(min_ms=58, onset=0.5, frame=0.3, fmax=4200.0, ghosts=False, trim=True, merge=0.03),
    "quiet_notes": dict(min_ms=110, onset=0.3, frame=0.2, fmax=3600.0, ghosts=True, trim=True),
}


def load_piece(folder, out):
    stems, truth = [], []
    for k in (1, 2):
        wav = sorted(folder.glob("AuSep_%d_vn_*.wav" % k))[0]
        y, _ = librosa.load(wav, sr=amt.SR, mono=True)
        stems.append(y)
        rows = np.loadtxt(sorted(folder.glob("Notes_%d_vn_*.txt" % k))[0], ndmin=2)
        for on, hz, dur in rows:
            truth.append(dict(start=float(on), end=float(on + dur), pitch=int(round(librosa.hz_to_midi(hz))), player=k - 1))
    n = min(len(s) for s in stems)
    stems = [s[:n] for s in stems]
    mix = stems[0] + stems[1]
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "mix.wav"), mix / (np.abs(mix).max() + 1e-9) * 0.9, amt.SR)
    truth.sort(key=lambda g: g["start"])
    for g in truth:  # loudness of this note relative to what the other player is doing, and context
        a, b = int(g["start"] * amt.SR), max(int(g["end"] * amt.SR), int(g["start"] * amt.SR) + 256)
        own = np.sqrt(np.mean(stems[g["player"]][a:b] ** 2) + 1e-12)
        oth = np.sqrt(np.mean(stems[1 - g["player"]][a:b] ** 2) + 1e-12)
        g["rel_db"] = float(20 * np.log10(own / oth))
        g["dur"] = g["end"] - g["start"]
        iv = [abs(g["pitch"] - h["pitch"]) for h in truth if h["player"] != g["player"] and min(h["end"], g["end"]) - max(h["start"], g["start"]) > 0.5 * min(g["dur"], h["end"] - h["start"])]
        g["with"] = "alone" if not iv else "unison" if 0 in iv else "octave" if any(i in (12, 24) for i in iv) else "other interval"
        g["upper"] = bool(iv) and any(h["pitch"] < g["pitch"] for h in truth if h["player"] != g["player"] and min(h["end"], g["end"]) > max(h["start"], g["start"]))
    return out / "mix.wav", truth


def match(truth, est):
    if not est or not truth:
        return []
    ri = np.array([[g["start"], g["end"]] for g in truth])
    rp = librosa.midi_to_hz(np.array([g["pitch"] for g in truth], float))
    ei = np.array([[n.start, max(n.end, n.start + 0.01)] for n in est])
    ep = librosa.midi_to_hz(np.array([n.pitch for n in est], float))
    return mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=TOL, pitch_tolerance=50.0, offset_ratio=None)


_DETECTED = {}
_AUDIO = {}


def run(mix, cfg):
    key = (str(mix), cfg["fmax"], cfg["onset"], cfg["frame"], cfg["min_ms"])
    if key not in _DETECTED:
        _DETECTED[key] = amt.poly_notes(mix, fmin=185.0, fmax=cfg["fmax"], onset=cfg["onset"], frame=cfg["frame"], min_ms=cfg["min_ms"])
    raw = copy.deepcopy(_DETECTED[key])
    if cfg.get("merge") is not None:
        raw = duet.merge_repeats(raw, cfg["merge"])
    if cfg.get("held"):
        if str(mix) not in _AUDIO:
            _AUDIO[str(mix)] = amt.load_mono(mix)
        raw = amt.merge_held(raw, _AUDIO[str(mix)])
    kept = duet.drop_overtone_ghosts(raw) if cfg["ghosts"] else list(raw)
    if cfg.get("blips"):
        kept = duet.drop_blips(kept)
    A, B = duet.split_two_voices(copy.deepcopy(kept), **cfg.get("split", {}))
    if cfg["trim"]:
        A, B = duet.trim_tails(A), duet.trim_tails(B)
    if cfg.get("leap"):
        A, B = duet.drop_leap_blips(A), duet.drop_leap_blips(B)
    if cfg.get("heal"):
        duet.heal_lines(A, B)
    return raw, A, B


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(VARIANTS)
    pieces = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("Notes_1_vn_*.txt")))
    results = {}
    for name in names:
        cfg = VARIANTS[name]
        tot = defaultdict(float)
        bucket = defaultdict(lambda: [0, 0])  # label -> [found, total]
        per_piece = {}
        for folder in pieces:
            mix, truth = load_piece(folder, out / folder.name)
            raw, A, B = run(mix, cfg)
            final = A + B
            label = {id(n): 0 for n in A} | {id(n): 1 for n in B}
            m_raw, m = match(truth, raw), match(truth, final)
            found = {r for r, _ in m}
            used = {e for _, e in m}
            tot["truth"] += len(truth)
            tot["raw_est"] += len(raw)
            tot["raw_hit"] += len(m_raw)
            tot["est"] += len(final)
            tot["hit"] += len(m)
            per_piece[folder.name] = dict(truth=len(truth), found=len(m), written=len(final))
            for i, g in enumerate(truth):
                hit = i in found
                keys = [
                    "length: under 0.15s" if g["dur"] < 0.15 else "length: 0.15-0.3s" if g["dur"] < 0.3 else "length: 0.3-0.6s" if g["dur"] < 0.6 else "length: over 0.6s",
                    "loudness: 10dB+ quieter than other" if g["rel_db"] < -10 else "loudness: 3-10dB quieter" if g["rel_db"] < -3 else "loudness: similar" if g["rel_db"] < 3 else "loudness: louder than other",
                    "other violin: " + g["with"] + (" (this is the upper note)" if g["with"] == "octave" and g["upper"] else " (this is the lower note)" if g["with"] == "octave" else ""),
                ]
                for k in keys:
                    bucket[k][1] += 1
                    bucket[k][0] += hit
                if not hit:  # was it written down, but an octave off?
                    if any(abs(n.start - g["start"]) <= TOL and abs(n.pitch - g["pitch"]) in (12, 24) for j, n in enumerate(final) if j not in used):
                        tot["missed_but_octave_off"] += 1
            for j, n in enumerate(final):
                if j in used:
                    continue
                tot["extra"] += 1
                under = [g for g in truth if g["start"] - TOL <= n.start <= g["end"]]
                if any(n.pitch - g["pitch"] in (12, 19, 24, 28) for g in under):
                    tot["extra_is_overtone"] += 1
                elif any(n.pitch - g["pitch"] in (-12, -24) for g in under):
                    tot["extra_is_octave_below"] += 1
                elif any(n.pitch == g["pitch"] for g in under):
                    tot["extra_is_repeat_of_held_note"] += 1
            agree = sum(1 for r, e in m if truth[r]["player"] == label[id(final[e])])
            tot["part_right"] += max(agree, len(m) - agree)
            for P in (A, B):  # these players never play two notes at once, so any chord written is false
                prof = amt.polyphony_profile(P)
                sounding = sum(n.dur for n in P)
                tot["part_time"] += sounding
                tot["false_chord_time"] += sounding * (1 - prof.get(1, 1.0))
        P, R = tot["hit"] / max(tot["est"], 1), tot["hit"] / max(tot["truth"], 1)
        results[name] = dict(
            settings=cfg,
            notes_in_truth=int(tot["truth"]),
            detector_only=dict(found=tot["raw_hit"] / tot["truth"], correct=tot["raw_hit"] / max(tot["raw_est"], 1)),
            final=dict(found=R, correct=P, f1=2 * P * R / max(P + R, 1e-9)),
            missed_notes=int(tot["truth"] - tot["hit"]),
            missed_but_written_an_octave_off=int(tot["missed_but_octave_off"]),
            extra_notes=dict(total=int(tot["extra"]), overtone_above_real_note=int(tot["extra_is_overtone"]), octave_below_real_note=int(tot["extra_is_octave_below"]), repeat_of_held_note=int(tot["extra_is_repeat_of_held_note"])),
            right_player=tot["part_right"] / max(tot["hit"], 1),
            false_chord_share=tot["false_chord_time"] / max(tot["part_time"], 1e-9),
            found_by_group={k: dict(found=v[0] / v[1], notes=v[1]) for k, v in sorted(bucket.items())},
            per_piece=per_piece,
        )
        r = results[name]
        print("\n== %s  %s" % (name, cfg))
        print("found %.1f%% of real notes, %.1f%% of written notes are real (F1 %.3f) | detector alone: found %.1f%%, real %.1f%%" % (100 * R, 100 * P, r["final"]["f1"], 100 * r["detector_only"]["found"], 100 * r["detector_only"]["correct"]))
        print("missed %d (of which written an octave off: %d) | extra %s | right player %.1f%% | false chords %.1f%% of part time" % (r["missed_notes"], r["missed_but_written_an_octave_off"], r["extra_notes"], 100 * r["right_player"], 100 * r["false_chord_share"]))
        for k, v in r["found_by_group"].items():
            print("   %-52s found %5.1f%%  of %d" % (k, 100 * v["found"], v["notes"]))
    (out / "eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
