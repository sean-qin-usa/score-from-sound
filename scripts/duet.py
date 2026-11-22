"""Transcribe two like-sounding instruments from one recording and split the notes into two parts.

usage: python -I scripts/duet.py <audio> <out_dir> [--fmin 185 --fmax 3600]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # isolated mode drops the script folder
import amt  # noqa: E402


def drop_overtone_ghosts(notes, tol=0.06):
    """Remove quiet notes that shadow a louder note exactly an octave/12th/2 octaves below."""
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if m is n or m.pitch >= n.pitch:
                continue
            if (n.pitch - m.pitch) in (12, 19, 24) and n.amp < 0.6 * m.amp:
                if m.start - tol <= n.start and n.end <= m.end + tol:
                    ghost = True
                    break
        if not ghost:
            keep.append(n)
    return keep


def drop_blips(notes, max_dur=0.15, tol=0.06):
    """Remove short notes sitting on an overtone of a louder note that is sounding at the same time.

    Unlike drop_overtone_ghosts this keeps sustained octave doublings: a real second violin an
    octave up holds its note; a detector ghost flickers."""
    keep = []
    for n in notes:
        ghost = False
        if n.dur < max_dur:
            for m in notes:
                if m is not n and (n.pitch - m.pitch) in (12, 19, 24, 28, 31) and m.amp >= n.amp:
                    if m.start - tol <= n.start and n.end <= m.end + tol:
                        ghost = True
                        break
        if not ghost:
            keep.append(n)
    return keep


def drop_leap_blips(part, max_dur=0.15, leap=9):
    """Within one part, remove a short note that jumps far above both the note before and the note after."""
    part = sorted(part, key=lambda n: (n.start, n.pitch))
    keep = []
    for i, n in enumerate(part):
        if n.dur < max_dur and 0 < i < len(part) - 1:
            if n.pitch - part[i - 1].pitch >= leap and n.pitch - part[i + 1].pitch >= leap:
                continue
        keep.append(n)
    return keep


def merge_repeats(notes, gap=0.03):
    """Join back-to-back notes of the same pitch: usually one held note the detector chopped up."""
    out, last = [], {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        p = last.get(n.pitch)
        if p is not None and n.start - p.end <= gap:
            p.end = max(p.end, n.end)
            p.amp = max(p.amp, n.amp)
        else:
            out.append(n)
            last[n.pitch] = n
    return out


def effective_durations(notes, gap=0.05):
    """Give each note the length of the same-pitch chain it sits in, so a held note that the
    detector chopped into pieces still counts as long."""
    chains = {}
    for n in sorted(notes, key=lambda n: n.start):
        c = chains.get(n.pitch)
        if c is not None and n.start - c[-1].end <= gap:
            c.append(n)
        else:
            c = chains[n.pitch] = [n]
        total = c[-1].end - c[0].start
        for m in c:
            m.eff = total
    for n in notes:
        n.eff = n.eff or n.dur


def split_two_voices(notes, beam=96, idle=0.0, dur_w=0.0, ring_ok=False, role_w=0.0, role_split=0.45, tone=None, tone_w=0.0, register_w=4.0):
    """Beam search over assignments: each part should move by small steps and rarely play two notes at once.

    idle   extra cost for handing a note to the part that is resting while the other part is the
           one playing (keeps a solo passage in one part instead of dividing it by register)
    dur_w  cost per doubling/halving of note length relative to the part's previous note
           (keeps a run of short notes together and a sustained line together)
    ring_ok  treat an overlap as a second string only if it also covers half of the new note
    register_w  cost, in semitones, of giving the higher of two sounding notes to the lower part
    tone     optional {id(note): (cost if player 0, cost if player 1)} from a tone classifier;
             with it part A is player 0 and part B is player 1
    role_w   split by role instead of by player: part A is the sustained line and part B the
             moving line. Cost per doubling that a note's held length is on the wrong side of
             `role_split` seconds for the part it is given to.
    """
    if role_w:
        effective_durations(notes)

    def step_cost(prev, n, other):
        resting = prev is None or n.start - prev.end > 1.0
        other_busy = other is not None and n.start - other.end < 0.4
        other_sounding = other is not None and other.end - n.start > 0.15
        cost = idle if (resting and other_busy and not other_sounding) else 0.0
        if prev is None:
            return cost
        gap = n.start - prev.end
        overlap = min(prev.end, n.end) - n.start
        if overlap > 0.15 and (not ring_ok or overlap > 0.5 * n.dur):  # else legato or room reverb
            cost += 5.0 + 14.0 * overlap / max(n.dur, 0.05)
        memory = 0.25 + 0.75 * math.exp(-max(gap, 0.0) / 1.5)  # a long rest loosens the link
        if dur_w:
            cost += dur_w * min(abs(math.log2(max(n.dur, 0.03) / max(prev.dur, 0.03))), 3.0)
        return cost + abs(n.pitch - prev.pitch) * memory

    # state: (cost, last note in part A, last note in part B, labels so far)
    states = [(0.0, None, None, ())]
    for n in notes:
        nxt = {}
        for cost, a, b, lab in states:
            for v, (pa, pb) in enumerate(((n, b), (a, n))):
                other = b if v == 0 else a
                c = cost + step_cost(a if v == 0 else b, n, other)
                if tone is not None and id(n) in tone:
                    c += tone_w * tone[id(n)][v]
                if role_w:
                    ratio = math.log2(max(n.eff, 0.03) / role_split)
                    c += role_w * min(max(-ratio if v == 0 else ratio, 0.0), 3.0)
                # mild preference for not swapping registers while both parts sound
                if other is not None and other.end > n.start + 0.04:
                    upper_is_a = (n.pitch > other.pitch) == (v == 0)
                    c += 0.0 if upper_is_a else register_w
                key = (id(pa), id(pb))
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, pa, pb, lab + (v,))
        states = sorted(nxt.values(), key=lambda s: s[0])[:beam]
    labels = states[0][3]
    A = [n for n, v in zip(notes, labels) if v == 0]
    B = [n for n, v in zip(notes, labels) if v == 1]
    if not role_w and tone is None and np.mean([n.pitch for n in A] or [0]) < np.mean([n.pitch for n in B] or [0]):
        A, B = B, A
    return A, B


def trim_tails(part, same_onset=0.05, tail=0.15):
    """Within one part, cut short ring-over into the next note, but keep real chords.

    Notes that start together are a chord. A note that keeps sounding well into the next one
    (more than `tail` seconds and more than half of that next note) is a held double stop and
    is kept; anything shorter is legato or room reverb and is cut at the next note's start."""
    part = sorted(part, key=lambda n: (n.start, n.pitch))
    for i, n in enumerate(part):
        for m in part[i + 1 :]:
            if m.start - n.start > same_onset:
                overlap = min(n.end, m.end) - m.start
                if 0 < overlap and not (overlap > tail and overlap > 0.5 * m.dur):
                    n.end = m.start
                break
    return [n for n in part if n.dur >= 0.06]


def split_parts(notes, audio_path, roles=False, use_stereo=True, min_spread_db=5.0, clean=False):
    """Divide the notes into two parts, choosing the evidence that suits the recording.

    A stereo recording whose notes are spread widely between left and right is split by side:
    each note goes to the part on its side of the recording's centre. On the test piece (two
    violins and piano, the violins about 7 dB apart) that put 83% of violin notes in the right
    part, where pitch movement managed 62% and any mixture of the two did worse, because the
    piano's notes pull the pitch-based logic around. One source alone spreads over roughly
    3-4 dB, so a spread of `min_spread_db` or more is taken to mean sources at different
    places; that bar comes from three recordings and is a starting point.

    `clean` says the notes are known to belong only to the instruments being divided (they
    came from a model that labels instruments). Then position is combined with pitch movement
    and register instead of deciding alone: notes clearly on one side were already right 95-99%
    of the time, but notes within 1.5 dB of the centre only 55%, and there the surrounding line
    should decide. On the test piece that raised the right-part share from 86.8% to 91.5%, with
    both halves of the piece agreeing; the position weight (12) was chosen on that one piece.
    With accompaniment notes still mixed in, the combination is worse than the plain cut.

    Otherwise (mono, or a narrow stereo image) the split is by pitch movement and register.
    Returns (part A, part B, what was used)."""
    if roles:
        A, B = split_two_voices(notes, role_w=3.0)
        return A, B, {"split_by": "sustained line / moving line"}
    pans = amt.pan_per_note(audio_path, notes) if use_stereo else None
    if pans is not None and len(notes) >= 40:
        spread = float(np.percentile(pans, 75) - np.percentile(pans, 25))
        centre = float(np.median(pans))
        if spread >= min_spread_db and clean:
            soft = lambda x: float(np.log1p(np.exp(np.clip(x, -30, 30))))
            evidence = {id(n): (soft((p - centre) / 2.7), soft(-(p - centre) / 2.7)) for n, p in zip(notes, pans)}
            A, B = split_two_voices(notes, tone=evidence, tone_w=12.0)
            return A, B, {"split_by": "left/right position with pitch movement and register", "spread_db": round(spread, 1), "centre_db": round(centre, 1)}
        if spread >= min_spread_db:
            A = [n for n, p in zip(notes, pans) if p < centre]
            B = [n for n, p in zip(notes, pans) if p >= centre]
            return A, B, {"split_by": "left/right position", "spread_db": round(spread, 1), "centre_db": round(centre, 1)}
        A, B = split_two_voices(notes)
        return A, B, {"split_by": "pitch movement and register", "stereo_spread_db": round(spread, 1), "why": "stereo image too narrow to separate players"}
    A, B = split_two_voices(notes)
    return A, B, {"split_by": "pitch movement and register", "why": "mono recording" if use_stereo else "left/right position switched off"}


def tone_disputes(y, A, B):
    """Second opinion on the split from tone alone, with no labels needed.

    Each violin's tone is learned from the notes where the split is safest (both parts sounding
    at once), then every note is asked which violin it sounds like. Returns the ids of notes
    where tone disagrees with the split, or None when there is too little overlap to learn from.
    On the URMP benchmark the split is right for about 90% of notes where the two opinions agree
    and about 71% where they disagree, so a disagreement marks a note worth checking by ear."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import tone_experiment as te

    S, freqs = te.spectrogram(y)
    notes = A + B
    part = {id(n): 0 for n in A} | {id(n): 1 for n in B}
    other = {0: B, 1: A}
    X = np.array([te.tone(S, freqs, n, [o for o in notes if o is not n and o.start < n.end and o.end > n.start]) for n in notes])
    lab = np.array([part[id(n)] for n in notes])
    safe = np.array([any(min(o.end, n.end) - max(o.start, n.start) > 0.5 * n.dur for o in other[part[id(n)]]) for n in notes])
    if safe.sum() < 40 or len(set(lab[safe])) < 2:
        return None
    Xs, mean = te.fill(X[safe])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xs, lab[safe])
    vote = clf.predict(te.fill(X, mean)[0])
    return {id(n) for n, v in zip(notes, vote) if v != part[id(n)]}


def split_voices(notes, n_parts, floors=None, register_w=4.0, beam=128, evidence=None, evidence_w=0.0):
    """Divide notes among `n_parts` players ordered from highest to lowest (a string quartet is
    violin 1, violin 2, viola, cello). The same evidence as the two-part split, generalised:

      - each part moves by small steps, and a long rest loosens that link;
      - a part rarely sounds two notes at once;
      - a part should not sit below a lower part's sounding note or above a higher part's;
      - `floors`: the lowest note each player's instrument can produce. A note under a part's
        floor cannot be given to it (instrument knowledge, not knowledge of the piece);
      - `evidence`: optional {id(note): cost per part} from something independent, such as
        left/right position.

    Returns a list of n_parts lists of notes."""
    floors = floors or [0] * n_parts

    def step_cost(prev, n):
        if prev is None:
            return 0.0
        gap = n.start - prev.end
        cost = 0.0
        overlap = min(prev.end, n.end) - n.start
        if overlap > 0.15:
            cost += 5.0 + 14.0 * overlap / max(n.dur, 0.05)
        memory = 0.25 + 0.75 * math.exp(-max(gap, 0.0) / 1.5)
        return cost + abs(n.pitch - prev.pitch) * memory

    states = [(0.0, (None,) * n_parts, ())]
    for n in sorted(notes, key=lambda x: (x.start, -x.pitch)):
        nxt = {}
        for cost, last, lab in states:
            for v in range(n_parts):
                if n.pitch < floors[v]:
                    continue
                c = cost + step_cost(last[v], n)
                for j, other in enumerate(last):
                    if j == v or other is None or other.end <= n.start + 0.04:
                        continue
                    if (j > v and n.pitch < other.pitch) or (j < v and n.pitch > other.pitch):
                        c += register_w  # this would put the parts out of order
                if evidence is not None and id(n) in evidence:
                    c += evidence_w * evidence[id(n)][v]
                new_last = last[:v] + (n,) + last[v + 1 :]
                key = tuple(id(x) for x in new_last)
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, new_last, lab + (v,))
        if not nxt:  # below every floor: give it to the lowest part
            nxt = {i: (cost, last, lab + (n_parts - 1,)) for i, (cost, last, lab) in enumerate(states)}
        states = sorted(nxt.values(), key=lambda z: z[0])[:beam]
    order = sorted(notes, key=lambda x: (x.start, -x.pitch))
    parts = [[] for _ in range(n_parts)]
    for n, v in zip(order, states[0][2]):
        parts[v].append(n)
    return parts


def stack_by_pitch(notes, n_parts, floors=None):
    """Baseline: at each note's start, rank it among the notes sounding then; the highest goes
    to part 1 and so on down. Knows nothing about lines, only about who is on top right now."""
    floors = floors or [0] * n_parts
    parts = [[] for _ in range(n_parts)]
    for n in notes:
        sounding = sorted({m.pitch for m in notes if m.start <= n.start + 0.04 and m.end > n.start + 0.04} | {n.pitch}, reverse=True)
        rank = sounding.index(n.pitch)
        # count from the bottom when few notes sound, so a lone low note goes to a low part
        v = min(rank, n_parts - 1) if len(sounding) >= n_parts else None
        if v is None:
            ok = [i for i in range(n_parts) if n.pitch >= floors[i]]
            v = ok[min(rank, len(ok) - 1)] if ok else n_parts - 1
        while v < n_parts - 1 and n.pitch < floors[v]:
            v += 1
        parts[v].append(n)
    return parts


def imitation_delay(notes, lo=0.5, hi=12.0, step=0.02, tol=0.08):
    """In a canon or round the second player repeats the first after a fixed delay. Returns
    (delay in seconds, share of note pairs that come back at that delay, the same share at a
    typical other delay), or None if there are too few notes.

    The test is on pairs of consecutive notes (this pitch, then that pitch, this far apart),
    not single pitches: one pitch recurs all the time, and a detector that writes a held note
    twice makes every note 'repeat' a fraction of a second later."""
    seq = sorted(notes, key=lambda n: (n.start, n.pitch))
    pairs = {}
    for i, a in enumerate(seq):
        for b in seq[i + 1 : i + 4]:
            gap = b.start - a.start
            if 0.05 < gap <= 0.6 and b.pitch != a.pitch:
                pairs.setdefault((a.pitch, b.pitch, round(gap / 0.06)), []).append(a.start)
    if len(pairs) < 20:
        return None
    lags = np.arange(lo, hi, step)
    score = np.zeros(len(lags))
    total = 0
    for ts in pairs.values():
        ts = np.array(ts)
        total += len(ts)
        d = (ts[None, :] - ts[:, None]).ravel()
        for x in d[(d > lo - tol) & (d < hi + tol)]:
            score[(lags > x - tol) & (lags < x + tol)] += 1
    k = int(np.argmax(score))
    return float(lags[k]), float(score[k] / max(total, 1)), float(np.median(score) / max(total, 1))


def split_canon(notes, delay, tol=0.12):
    """Divide a canon at the unison: a note that comes back `delay` seconds later belongs to the
    leader, a note that was heard `delay` seconds earlier to the follower. Notes for which both
    or neither hold go to whichever line has the nearest decided note in pitch and time.
    Returns (leader notes, follower notes, share of notes the delay decided on its own)."""
    by = {}
    for n in notes:
        by.setdefault(n.pitch, []).append(n.start)
    by = {p: np.array(ts) for p, ts in by.items()}
    lead, follow, unsure = [], [], []
    for n in notes:
        later = bool(np.any(np.abs(by[n.pitch] - (n.start + delay)) <= tol))
        earlier = bool(np.any(np.abs(by[n.pitch] - (n.start - delay)) <= tol))
        (lead if later and not earlier else follow if earlier and not later else unsure).append(n)
    decided = (len(lead) + len(follow)) / max(len(notes), 1)
    A, B = list(lead), list(follow)

    def distance(n, line):
        return min((abs(m.pitch - n.pitch) + 4 * abs(m.start - n.start) for m in line if abs(m.start - n.start) < 1.0), default=99.0)

    for n in unsure:
        (A if distance(n, lead) <= distance(n, follow) else B).append(n)
    return sorted(A, key=lambda n: n.start), sorted(B, key=lambda n: n.start), decided


def crossings(A, B, step=0.05):
    end = max(n.end for n in A + B)
    grid = np.arange(0, end, step)

    def line(notes):
        p = np.full(len(grid), np.nan)
        for n in notes:
            p[int(n.start / step) : int(n.end / step)] = n.pitch
        return p

    a, b = line(A), line(B)
    both = np.isfinite(a) & np.isfinite(b)
    sign = np.sign(a[both] - b[both])
    sign = sign[sign != 0]
    return int((np.diff(sign) != 0).sum()), float((sign < 0).mean()) if len(sign) else 0.0, float(both.mean())


FAMILIES = {"piano": (0, 8), "strings": (40, 56), "brass": (56, 64), "reed": (64, 72), "pipe": (72, 80), "guitar": (24, 32), "bass": (32, 40)}


def note_number(name):
    """MIDI number of a note name such as G3, C#4 or Bb2 (middle C is C4 = 60)."""
    steps = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    letter, rest = name[0].upper(), name[1:]
    shift = 1 if rest.startswith("#") else -1 if rest.startswith("b") else 0
    return 12 * (int(rest.lstrip("#b")) + 1) + steps[letter] + shift


def notes_from_midi(path, family):
    """(notes of the chosen family, every other pitched note, {track name: note count})."""
    import pretty_midi

    lo, hi = FAMILIES[family]
    chosen, rest, labels = [], [], {}
    for inst in pretty_midi.PrettyMIDI(str(path)).instruments:
        if inst.is_drum:
            continue
        name = pretty_midi.program_to_instrument_name(inst.program)
        labels[name] = labels.get(name, 0) + len(inst.notes)
        target = chosen if lo <= inst.program < hi else rest
        target += [amt.Note(n.start, n.end, n.pitch, n.velocity / 127) for n in inst.notes]
    return sorted(chosen, key=lambda n: n.start), sorted(rest, key=lambda n: n.start), labels


def cross_check(notes, others, tol=0.15):
    """Mark which notes a second, independent note source also wrote; return the ones only it wrote.

    Two detectors built differently and trained on different recordings rarely invent the same
    note, so a note both wrote is far more likely to be real than a note only one wrote (on a
    string quartet neither had heard: 96% against 70% and 48%). Nothing about the instruments
    is assumed. Notes pair off one to one: same pitch, starts within `tol` seconds, nearest
    first. Pairing by "the other source has this pitch sounding" was tried and confirms the
    extra pieces of a held note that one source wrote in fragments, which are mostly wrong.
    Sets note.seen to 1.0 (both) or 0.5 (one) on both lists."""
    by_pitch = {}
    for o in others:
        o.seen = 0.5
        by_pitch.setdefault(o.pitch, []).append(o)
    pairs = []
    for n in notes:
        n.seen = 0.5
        pairs += [(abs(o.start - n.start), id(n), id(o), n, o) for o in by_pitch.get(n.pitch, []) if abs(o.start - n.start) <= tol]
    for _, _, _, n, o in sorted(pairs, key=lambda p: p[:3]):
        if n.seen < 1.0 and o.seen < 1.0:
            n.seen = o.seen = 1.0
    return [o for o in others if o.seen < 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--fmin", type=float, default=185.0)
    ap.add_argument("--fmax", type=float, default=4200.0)
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--min-ms", type=float, default=58.0, help="shortest note to keep")
    ap.add_argument("--canon", nargs="?", const="auto", default=None, metavar="SECONDS", help="the piece is a canon or round at the same pitch: divide the players by the delay between them (found from the notes, or given in seconds). Not automatic, because ordinary accompaniment figures repeat too")
    ap.add_argument("--notes-from", default=None, help="take the notes from this MIDI file (for example one written by scripts/run_yourmt3.py) instead of running the built-in detector")
    ap.add_argument("--cross-check", action="store_true", help="with --notes-from: also run the built-in detector and mark every note only one of the two wrote as unsure. When the model hears one family almost alone (9 notes in 10), the notes only the built-in detector wrote, and the notes the model gave to some other instrument, are added as unsure notes too")
    ap.add_argument("--join-held", action="store_true", help="with --notes-from: also join a held note written twice (off by default; it cost 2 points of notes found on both recordings that have an answer key)")
    ap.add_argument("--lowest", default=None, help="lowest note these instruments can play, e.g. G3 for violins; anything below it cannot be theirs and goes to the accompaniment")
    ap.add_argument("--family", default="strings", choices=sorted(FAMILIES), help="which instrument family in that MIDI file holds the two lines to split")
    ap.add_argument("--no-stereo", action="store_true", help="ignore left/right position even when the recording is stereo")
    ap.add_argument("--check-overtones", action="store_true", help="remove notes the recording says are only an overtone of a lower note (for pieces with piano)")
    ap.add_argument("--keep-rewrites", action="store_true", help="do not join a held note that the detector wrote twice")
    ap.add_argument("--solo", default="", help="time ranges where only one violin plays, e.g. 0-13,95-102 (seconds)")
    ap.add_argument("--roles", action="store_true", help="split into a sustained line and a moving line instead of guessing which player is which")
    ap.add_argument("--keep-leaps", action="store_true", help="keep short notes that jump far above their neighbours")
    ap.add_argument("--drop-blips", action="store_true", help="remove short notes on an overtone of a louder note (also hides some real octave doublings)")
    ap.add_argument("--drop-overtones", action="store_true", help="remove quiet notes an octave above a louder one (also removes real octave doublings)")
    ap.add_argument("--frame", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y = amt.load_mono(args.audio)
    accompaniment = []
    if args.notes_from:
        # An instrument-aware model has already said which notes belong to which instrument.
        # Its notes are used as they are: on the two-violin benchmark the held-note check took
        # 2.3 points off the notes found for half a point of accuracy, so it is not applied.
        raw, accompaniment, labels = notes_from_midi(args.notes_from, args.family)
        notes = list(raw)
        source = {"notes_from": Path(args.notes_from).name, "family": args.family, "tracks_in_file": labels}
        if args.lowest:
            floor = note_number(args.lowest)
            below = [n for n in notes if n.pitch < floor]
            notes = [n for n in notes if n.pitch >= floor]
            accompaniment = sorted(accompaniment + below, key=lambda n: n.start)
            source["below_%s_moved_to_accompaniment" % args.lowest] = len(below)
        if args.join_held:
            before = len(notes)
            notes = amt.merge_held(notes, y)
            source["held_notes_joined"] = before - len(notes)
        if args.cross_check:
            plain = amt.merge_held(amt.poly_notes(args.audio, fmin=args.fmin, fmax=args.fmax, onset=args.onset, frame=args.frame, min_ms=args.min_ms), y)
            if args.lowest:
                plain = [n for n in plain if n.pitch >= floor]
            in_range = [n for n in accompaniment if not args.lowest or n.pitch >= floor]
            one_family = len(notes) >= 0.9 * (len(notes) + len(accompaniment))
            only_plain = cross_check(notes + (in_range if one_family else []), plain)
            source["cross_check"] = {"written_by_both": sum(n.seen >= 1.0 for n in notes), "written_by_the_model_only": sum(n.seen < 1.0 for n in notes), "one_family_recording": bool(one_family)}
            if one_family:
                # nothing else is playing, so a note either source wrote can only belong to this family
                strays = {id(n) for n in in_range}
                for n in in_range:
                    n.seen = min(n.seen, 0.5)  # the model itself doubted it
                source["cross_check"].update(added_from_built_in_detector=len(only_plain), returned_from_other_instrument_tracks=len(in_range))
                notes = sorted(notes + in_range + only_plain, key=lambda n: n.start)
                accompaniment = [n for n in accompaniment if id(n) not in strays]
    else:
        raw = amt.poly_notes(args.audio, fmin=args.fmin, fmax=args.fmax, onset=args.onset, frame=args.frame, min_ms=args.min_ms)
        notes = drop_overtone_ghosts(raw) if args.drop_overtones else list(raw)
        if not args.keep_rewrites:
            notes = amt.merge_held(notes, y)
        if args.check_overtones:
            notes = amt.drop_checked_overtones(notes, y)
        if args.drop_blips:
            notes = drop_blips(notes)
        source = {"notes_from": "built-in detector"}
    if args.canon:
        if args.canon == "auto":
            found = imitation_delay(notes, lo=1.0)
            if found is None:
                raise SystemExit("too few notes to find a canon delay; give it in seconds with --canon")
            delay, strength, typical = found
        else:
            delay, strength, typical = float(args.canon), None, None
        A, B, decided = split_canon(notes, delay)
        stereo = {"split_by": "canon delay", "delay_sec": round(delay, 2), "decided_by_delay_alone": round(decided, 2), "note_pairs_returning_at_delay": strength, "at_a_typical_delay": typical}
    else:
        A, B, stereo = split_parts(notes, args.audio, roles=args.roles, use_stereo=not args.no_stereo, clean=bool(args.notes_from))
    for span in filter(None, "" if args.roles else args.solo.split(",")):  # a hint from someone who knows the piece
        lo, hi = (float(x) for x in span.split("-"))
        inA = [n for n in A if lo <= n.start < hi]
        inB = [n for n in B if lo <= n.start < hi]
        if len(inA) >= len(inB):
            A, B = A + inB, [n for n in B if not lo <= n.start < hi]
        else:
            A, B = [n for n in A if not lo <= n.start < hi], B + inA
    if args.roles:
        A = merge_repeats(A, gap=0.05)  # in the sustained line, back-to-back same pitches are one held note
    A, B = trim_tails(A), trim_tails(B)
    if not args.keep_leaps:
        A, B = drop_leap_blips(A), drop_leap_blips(B)
    if not args.roles and not args.canon and np.mean([n.pitch for n in A] or [0]) < np.mean([n.pitch for n in B] or [0]):
        A, B = B, A
    names = ("Sustained line", "Moving line") if args.roles else ("Leader", "Follower") if args.canon else ("Violin 1", "Violin 2")
    # the tone check was validated against the pitch-based split; after a split by side it
    # disagrees with nearly half the notes and tells you nothing, so it is skipped there
    disputed = None if args.roles or args.canon or stereo.get("split_by", "").startswith("left/right position") else tone_disputes(y, A, B)
    for n in A + B:
        n.conf = 0.0 if disputed and id(n) in disputed else 1.0
    beats, bpm, jitter = amt.beat_grid(y)
    n_cross, frac_inverted, frac_both = crossings(A, B)

    report = {
        "note_source": source,
        "accompaniment_notes": len(accompaniment),
        "notes_raw": len(raw),
        "notes_after_joining_held_notes": len(notes),
        "part1_notes": len(A),
        "part2_notes": len(B),
        "part1_range": [amt.name(min(n.pitch for n in A)), amt.name(max(n.pitch for n in A))],
        "part2_range": [amt.name(min(n.pitch for n in B)), amt.name(max(n.pitch for n in B))],
        "simultaneous_notes_share_after_cleanup": amt.polyphony_profile(notes),
        "part1_double_stop_share": amt.polyphony_profile(A).get(2, 0.0),
        "part2_double_stop_share": amt.polyphony_profile(B).get(2, 0.0),
        "share_of_time_both_parts_play": frac_both,
        "part_crossings": n_cross,
        "share_of_duet_time_part2_above_part1": frac_inverted,
        "chroma_agreement_with_audio": amt.chroma_agreement(A + B, y),
        "how_parts_were_split": stereo,
        "notes_where_tone_disputes_the_split": None if disputed is None else len(disputed) / max(len(A) + len(B), 1),
        "notes_only_one_source_wrote": sum(n.seen < 1.0 for n in A + B) / max(len(A) + len(B), 1) if args.cross_check else None,
        "tempo_bpm": bpm,
        "beat_jitter": jitter,
    }
    print(json.dumps(report, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))

    import soundfile as sf

    amt.write_check(out / "check_duet.wav", y, A + B)  # recording left, transcription right
    pa, pb = amt.synth(A, len(y)), amt.synth(B, len(y))
    peak = max(np.abs(pa).max(), np.abs(pb).max(), 1e-9)
    sf.write(str(out / "check_parts.wav"), np.stack([pa, pb], axis=1) / peak * 0.7, amt.SR)  # part 1 left, part 2 right

    # real-time MIDI keeps the performance timing; the gridded copy is for notation
    extra = [("Accompaniment (as labelled by the model)", 0, accompaniment)] if accompaniment else []
    amt.write_midi(out / "duet_performance.mid", [(names[0], 40, A), (names[1], 40, B)] + extra, bpm)
    gA, gB = amt.warp(A, beats, bpm, grid=4), amt.warp(B, beats, bpm, grid=4)
    gAcc = amt.warp(accompaniment, beats, bpm, grid=4)
    amt.write_midi(out / "duet_gridded.mid", [(names[0], 40, gA), (names[1], 40, gB)] + ([("Accompaniment (as labelled by the model)", 0, gAcc)] if gAcc else []), bpm)
    staves = [(names[0], "treble", gA), (names[1], "treble", gB)]
    if gAcc:  # a keyboard part on two staves, divided at middle C
        staves += [("Accompaniment, upper staff", "treble", [n for n in gAcc if n.pitch >= 60]), ("Accompaniment, lower staff", "bass", [n for n in gAcc if n.pitch < 60])]
    amt.write_musicxml(out / "duet.musicxml", staves, bpm, unsure_below=0.5, unsure_style="color")


if __name__ == "__main__":
    main()
