"""Reassemble per-page MusicXML from the oemer score reader into one score with named staves.

The reader writes a system of N staves as blocks, one after the other, each block starting
with a <print> marker. Two shapes occur:

  paired      the reader grouped the staves in twos (it found barlines joining them): each
              block holds two staves, and a system is N/2 blocks.
  one-by-one  the reader could not group them (systems that are one long measure): each block
              is a single staff. Such a page is used only if it was read from an image with
              helper barlines (scripts/helper_barlines.py) and the number of blocks equals
              the number of staves found on that image, so each block can be placed by its
              position on the page. Anything else is skipped and listed, never guessed.

usage: python -I scripts/omr_stitch.py <dir with page-NN.musicxml> <out_dir>
           [--layout "Violin 1,Violin 2,Piano,Piano"] [--beats 4] [--last-page N]
           [--helped <dir of reads from helper-barline images>] [--page-info <dir of page-NN.png.json>]
--layout lists the staves of one system from top to bottom; staves with the same name are
merged into one part (a piano's two staves).

Key signatures are then put right by the rules a player reads by (--no-key-repair turns this
off). The reader keeps one running key signature and changes it whenever a measure opens with
sharps or flats in every staff it is looking at, which goes wrong in three ways:

  courtesy signature   the new signature printed after the last barline of a line, announcing
                       a change, is read as one more measure. A last "measure" that holds a
                       signature and no notes in any staff of the system is not a measure.
  staves disagree      all staves of a system carry the same signature. Where the reader's
                       count differs between them (four flats above, five below), the one
                       that continues the line before wins, or after an announced change the
                       one that differs from it.
  change inside a line  counts only if every staff of the system shows it at the same
                       measure, and takes its value from the clean signature at the start of
                       the next line (the cancelling naturals printed at the change spoil
                       the count there). A change the next line does not confirm, or that
                       only some staves show, was accidentals on the first notes: they
                       hold to the barline, so that measure is kept as read and the
                       measures after it go back to the signature in force.

Notes are then respelled: a note whose sharp or flat came from the misread signature gets
what the corrected signature gives that letter; a note carrying its own accidental keeps it.
Only the notation is used, never a recording, so a recording can test the result.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pretty_midi
from music21 import converter

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def load_page(path):
    """Staves (each a list of measures in document order) and blocks of measure positions."""
    text = Path(path).read_text()
    part = re.split(r"<part id=", text)[1]
    bodies = re.findall(r'<measure number="\d+"[^>]*>(.*?)</measure>', part, flags=re.S)
    blocks = []
    for i, body in enumerate(bodies):
        if "<print" in body or not blocks:
            blocks.append([])
        blocks[-1].append(i)
    staves = [list(p.getElementsByClass("Measure")) for p in converter.parse(str(path)).parts]
    keys, state = [], 0
    for body in bodies:  # the reader's running key signature, and whether the measure holds any notes
        found = re.search(r"<fifths>(-?\d+)</fifths>", body)
        if found:
            state = int(found.group(1))
        keys.append({"read": state, "changed": bool(found), "pitched": len(re.findall(r"<pitch>", body))})
    return staves, blocks, keys


def contents(measure):
    """Notes of one measure as (offset, length, midi pitch), and the measure's filled length."""
    notes, filled = [], 0.0
    if measure is None:
        return notes, filled
    for el in measure.flatten().notesAndRests:
        if el.duration.isGrace or el.quarterLength == 0:
            continue
        filled = max(filled, float(el.offset + el.quarterLength))
        if el.isRest:
            continue
        for p in el.pitches:
            notes.append((float(el.offset), float(el.quarterLength), int(p.midi), p.step, int(p.alter)))
    return notes, filled


SHARPS, FLATS = "FCGDAEB", "BEADGCF"


def from_signature(fifths, letter):
    """What a key signature of `fifths` sharps (negative: flats) does to a letter: +1, -1 or 0."""
    if fifths > 0 and letter in SHARPS[:fifths]:
        return 1
    if fifths < 0 and letter in FLATS[:-fifths]:
        return -1
    return 0


def courtesy_measures(system, keys):
    """How many trailing 'measures' of a system are a courtesy signature: a signature and no notes."""
    n = 0
    while len(system[0]) - n > 1:
        last = [keys[b[-1 - n]] for b in system]
        if any(k["changed"] for k in last) and not any(k["pitched"] for k in last):
            n += 1
        else:
            break
    return n


def fit_bar(notes, beats):
    """A staff's bar whose notes run past the barline: the reader misjudged some lengths, and
    every note after a misjudged one is displaced. Stretch the bar's notes to fit (a note at
    4.5 beats in a 4/4 bar goes to 4). Tested against the recording on a fifteen-page score:
    the notes this moved went from 45% to 51% found by the instrument-aware model, and the
    alternative, finding one length to correct, reached 49%."""
    if not notes or max(off for off, *_ in notes) < beats - 1e-6:
        return notes, None
    span = max(off + dur for off, dur, *_ in notes)
    scale = beats / span
    return [(off * scale, dur * scale, *rest) for off, dur, *rest in notes], scale


def systems_paired(staves, blocks, n_staves, beats, helper=None, keys=None, rhythm=True):
    """Each block is two staves; a system is n_staves/2 blocks with equal measure counts.
    `helper` lists, per system, whether a helper barline split its single measure in two; those
    two halves are laid end to end and squeezed back into one measure."""
    per = -(-n_staves // 2)
    if len(staves) != 2 or len(blocks) % per:
        return None
    out = []
    for i in range(0, len(blocks), per):
        system = blocks[i : i + per]
        if len({len(b) for b in system}) != 1:
            return None
        whole = system
        dropped = courtesy_measures(system, keys) if keys else 0
        if dropped:
            system = [b[:-dropped] for b in system]
        notes = []
        merge = bool(helper and len(helper) == len(blocks) // per and helper[i // per] and len(system[0]) == 2)
        fitted = []
        for b, block in enumerate(system):
            for s in range(2):
                k = 2 * b + s
                if k >= n_staves:
                    continue
                got = [contents(staves[s][idx] if idx < len(staves[s]) else None) for idx in block]
                if merge:
                    (n1, f1), (n2, f2) = got
                    total = (f1 + f2) or beats
                    for half, (base, ns) in enumerate(((0.0, n1), (f1, n2))):
                        notes += [(k, 0, min((base + off) * beats / total, beats - 1e-6), dur * beats / total, pitch, step, alter, block[half], half) for off, dur, pitch, step, alter in ns]
                else:
                    for j, (ns, _) in enumerate(got):
                        if rhythm:
                            ns, scale = fit_bar(ns, beats)
                            if scale is not None:
                                fitted.append({"staff": k, "measure_in_system": j + 1, "stretched_by": round(scale, 3)})
                        notes += [(k, j, min(off, beats), dur, pitch, step, alter, block[j], j) for off, dur, pitch, step, alter in ns]  # an over-full bar keeps its notes
        out.append({"measures": 1 if merge else len(system[0]), "notes": notes, "how": "paired", "blocks": system, "courtesy": [whole[0][-1 - d] for d in range(dropped)], "lined_up": True, "fitted": fitted})
    return out


def systems_one_by_one(staves, blocks, n_staves, beats, real_measures):
    """Each block is one staff, in page order. A staff's measures are laid end to end and
    stretched to the system's true length, so a misread rhythm in one staff cannot push it
    out of step with the others beyond the end of the system."""
    if len(staves) != 1 or len(blocks) % n_staves or len(blocks) // n_staves != len(real_measures):
        return None
    out = []
    for i in range(0, len(blocks), n_staves):
        target = real_measures[i // n_staves] * beats
        notes = []
        for k, block in enumerate(blocks[i : i + n_staves]):
            pos, got = 0.0, []
            for idx in block:
                ns, filled = contents(staves[0][idx])
                got += [(pos + off, dur, pitch, step, alter, idx) for off, dur, pitch, step, alter in ns]
                pos += filled
            scale = target / pos if pos > 0 else 1.0
            for off, dur, pitch, step, alter, idx in got:
                pos = min(off * scale, target - 1e-6)
                notes.append((k, int(pos // beats), pos % beats, dur * scale, pitch, step, alter, idx, 0))
        out.append({"measures": real_measures[i // n_staves], "notes": notes, "how": "one-by-one", "blocks": blocks[i : i + n_staves], "courtesy": [], "lined_up": False})
    return out


def repair_keys(systems):
    """Decide the key signature of every measure of every system, in order through the piece.

    systems: dicts with "blocks" (measure indices per group of staves), "keys" (the page's
    per-measure reading), "courtesy" and "lined_up" (the groups' measures correspond one to
    one). Adds "key_at" (signature per measure position) to each and returns a log."""
    from collections import Counter

    def starts(sy):
        return [sy["keys"][b[0]]["read"] for b in sy["blocks"]]

    log, before, announced = [], None, False
    for n, sy in enumerate(systems):
        cands = starts(sy)
        after = starts(systems[n + 1]) if n + 1 < len(systems) else []
        after = after[0] if after and len(set(after)) == 1 else None
        if len(set(cands)) == 1:
            start = cands[0]
        else:
            pool = [c for c in cands if c != before] if announced else [c for c in cands if c == before]
            pool = pool or ([c for c in cands if c == after] or cands)
            start = Counter(pool).most_common(1)[0][0]
            log.append({"page": sy["page"], "system": sy["number"], "rule": "staves disagree", "read": cands, "taken": start})
        length = len(sy["blocks"][0])
        key_at = [start] * length
        if sy["lined_up"]:
            for j in range(1, length):
                here = [sy["keys"][b[j]] for b in sy["blocks"]]
                if not any(k["changed"] for k in here):
                    continue
                read = [k["read"] for k in here if k["changed"]]
                if all(k["changed"] for k in here):
                    value = after if after is not None else Counter(read).most_common(1)[0][0]
                    if value != key_at[j]:
                        log.append({"page": sy["page"], "system": sy["number"], "measure_in_system": j + 1, "rule": "change inside a line", "read": read, "taken": value})
                        key_at[j:] = [value] * (length - j)
                        continue
                    rule = "change the next line does not confirm"
                else:
                    rule = "change in only some staves"
                # accidentals on the first notes, which hold to the barline: this measure stays as read
                log.append({"page": sy["page"], "system": sy["number"], "measure_in_system": j + 1, "rule": rule, "read": read, "taken": "this measure as read, then %d again" % key_at[j]})
                key_at[j] = None
        if sy["courtesy"]:
            log.append({"page": sy["page"], "system": sy["number"], "rule": "courtesy signature", "read": [sy["keys"][i]["read"] for i in sy["courtesy"]], "measures_removed": len(sy["courtesy"])})
        sy["key_at"] = key_at
        before, announced = [k for k in key_at if k is not None][-1], bool(sy["courtesy"])
    return log


def stitch(pages_dir, layout, beats, last_page=None, helped_dir=None, info_dir=None, repair=True, rhythm=True):
    n = len(layout)
    rows, skipped, page_info, fill = [], [], [], []
    measure_index = 0
    gathered = []
    for f in sorted(Path(pages_dir).glob("page-*.musicxml")):
        if last_page is not None and int(re.search(r"(\d+)", f.stem).group(1)) > last_page:
            continue
        candidates = [("as read", f)]
        if helped_dir and (Path(helped_dir) / f.name).exists():
            candidates.append(("with helper barlines", Path(helped_dir) / f.name))
        info_path = Path(info_dir) / (f.stem + ".png.json") if info_dir else None
        info = json.loads(info_path.read_text()) if info_path and info_path.exists() else None
        if info and len(candidates) == 2 and any(s.get("helper") for s in info["systems"]):
            candidates.reverse()  # the page needed a helper barline, so trust that read first
        systems, used, tried = None, None, []
        for label, path in candidates:
            staves, blocks, keys = load_page(path)
            helper = [bool(x.get("helper")) for x in info["systems"]] if info and label == "with helper barlines" else None
            systems = systems_paired(staves, blocks, n, beats, helper, keys if repair else None, rhythm)
            if systems is None and info and label == "with helper barlines" and len(blocks) == info["staves_found"]:
                real = [s.get("interior_barlines", 0) + 1 for s in info["systems"]]
                systems = systems_one_by_one(staves, blocks, n, beats, real)
            tried.append({"read": label, "staves_in_file": len(staves), "blocks": len(blocks)})
            if systems is not None:
                used = label
                break
        if systems is None:
            skipped.append({"page": f.name, "tried": tried, "staves_on_page": info["staves_found"] if info else None})
            continue
        for number, system in enumerate(systems):
            system.update(page=f.name, number=number + 1, keys=keys)
        gathered += systems
        page_info.append({"page": f.name, "read": used, "how": systems[0]["how"], "systems": len(systems), "measures_per_system": [s["measures"] for s in systems]})
    key_log = repair_keys(gathered) if repair else []
    respelled = {}
    key_per_measure, last = [], 0
    for system in gathered:
        signatures = system.get("key_at") or [system["keys"][i]["read"] for i in system["blocks"][0]]
        for j in range(system["measures"]):
            k = signatures[min(j, len(signatures) - 1)]
            last = last if k is None else k
            key_per_measure.append(last)
        for k, j, off, dur, pitch, step, alter, idx, position in system["notes"]:
            if repair:
                read, right = system["keys"][idx]["read"], system["key_at"][min(position, len(system["key_at"]) - 1)]
                if right is not None and read != right and alter == from_signature(read, step) and from_signature(right, step) != alter:
                    pitch += from_signature(right, step) - alter
                    respelled[system["page"]] = respelled.get(system["page"], 0) + 1
            rows.append(dict(part=layout[k], staff=k, measure=measure_index + j + 1, beat=round(off + 1, 4), duration_beats=round(dur, 4), pitch_midi=pitch, pitch_name="%s%d" % (NAMES[pitch % 12], pitch // 12 - 1), page=system["page"]))
        per_measure = {}
        for k, j, off, dur, *_ in system["notes"]:
            per_measure[(k, j)] = max(per_measure.get((k, j), 0.0), off + dur)
        fill += [v / beats for v in per_measure.values()]
        measure_index += system["measures"]
    f = np.array(fill) if fill else np.array([0.0])
    check = {"measures": measure_index, "staff_measures_filling_the_bar_within_10pct": float(np.mean(np.abs(f - 1) < 0.1)), "pages_skipped": skipped, "key_signature_per_measure": key_per_measure}
    if rhythm:
        bars = [dict(page=sy["page"], system=sy["number"], **x) for sy in gathered for x in sy.get("fitted", [])]
        check["rhythm_repair"] = {"staff_bars_stretched_to_fit": len(bars), "bars": bars}
    if repair:
        check["key_signature_repair"] = {"notes_respelled": sum(respelled.values()), "notes_respelled_by_page": respelled, "courtesy_measures_removed": sum(e.get("measures_removed", 0) for e in key_log), "decisions": key_log}
    return rows, page_info, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--layout", default="Violin 1,Violin 2,Piano,Piano")
    ap.add_argument("--beats", type=float, default=4.0, help="quarter notes per measure")
    ap.add_argument("--last-page", type=int, default=None, help="join pages 1..N only")
    ap.add_argument("--helped", default=None, help="folder of reads made from helper-barline images")
    ap.add_argument("--page-info", default=None, help="folder of page-NN.png.json files written by helper_barlines.py")
    ap.add_argument("--no-key-repair", action="store_true", help="keep the key signatures exactly as the reader gave them")
    ap.add_argument("--no-rhythm-repair", action="store_true", help="keep the reader's rhythms even where a bar's notes run past the barline")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = [x.strip() for x in args.layout.split(",")]
    rows, page_info, check = stitch(args.pages, layout, args.beats, args.last_page, args.helped, args.page_info, repair=not args.no_key_repair, rhythm=not args.no_rhythm_repair)
    check["covers_pages"] = [1, args.last_page] if args.last_page else "all that could be joined"

    with open(out / "notes_score.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "staff", "measure", "beat", "duration_beats", "pitch_midi", "pitch_name", "page"])
        w.writeheader()
        w.writerows(rows)
    pm = pretty_midi.PrettyMIDI(initial_tempo=60)  # one quarter note = one second; tempo is nominal
    for name in dict.fromkeys(layout):
        inst = pretty_midi.Instrument(program=0 if "piano" in name.lower() else 40, name=name)
        for r in rows:
            if r["part"] == name:
                s = (r["measure"] - 1) * args.beats + r["beat"] - 1
                inst.notes.append(pretty_midi.Note(80, r["pitch_midi"], s, s + max(r["duration_beats"], 0.05)))
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments.append(inst)
    pm.write(str(out / "score.mid"))
    counts = {name: sum(r["part"] == name for r in rows) for name in dict.fromkeys(layout)}
    meta = {"layout": layout, "beats_per_measure": args.beats, "notes_per_part": counts, "pages": page_info, **check}
    (out / "score_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("pages", "key_signature_per_measure", "rhythm_repair", "key_signature_repair")}, indent=1))


if __name__ == "__main__":
    main()
