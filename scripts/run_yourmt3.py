"""Run the YourMT3+ multi-instrument transcription model on a recording.

YourMT3+ (Chang et al., 2024; Apache-2.0) writes a MIDI file with one track per instrument it
believes it hears, so unlike the note detector used elsewhere here it says which instrument
plays each note. This wrapper expects the model's inference code and one checkpoint under
tools/yourmt3/ (the "YourMT3" Space and model repository of the authors on Hugging Face) and
its own environment (.venv_mt3).

The model listens in windows of about two seconds, and a note that crosses the edge of a window
is often cut in two, started late or lost. So the recording is transcribed twice, the second
time with the window edges moved by half a window, and the two are joined: a note both passes
wrote is taken from the pass whose window edge is farther from it, and a note only one pass
wrote is kept when it lies farther from that pass's edges than from the other's. On six
recordings with a score this raised both recall and precision on every one (F from 74.4 to
75.7 on average, most on the Mozart quartet and the Telemann). It doubles the running time;
--single runs one pass.

usage: .venv_mt3/bin/python scripts/run_yourmt3.py [--single | --shift SECONDS] <recording> <out.mid> [...]
--shift runs one pass with the windows moved by that many seconds (for experiments).
"""
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "yourmt3"
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
        "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
        "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1"]


def edge_distance(t, offset, window):
    """Seconds from time t to the nearest window edge of a pass whose edges sit at offset + k*window."""
    x = (t - offset) % window
    return min(x, window - x)


def stitch(first, second, window, shift, tol=0.1):
    """Join two passes whose window edges are `shift` seconds apart (see the note at the top)."""
    offsets = (0.0, -shift)
    by_pitch = {}
    for j, m in enumerate(second):
        by_pitch.setdefault((m.is_drum, m.pitch), []).append(j)
    close = sorted((abs(second[j].onset - n.onset), i, j) for i, n in enumerate(first) for j in by_pitch.get((n.is_drum, n.pitch), []) if abs(second[j].onset - n.onset) <= tol)
    pair, taken = {}, set()
    for _, i, j in close:
        if i not in pair and j not in taken:
            pair[i] = j
            taken.add(j)
    out = []
    for i, n in enumerate(first):
        if i in pair:
            m = second[pair[i]]
            out.append(n if edge_distance(n.onset, offsets[0], window) >= edge_distance(m.onset, offsets[1], window) else m)
        elif edge_distance(n.onset, offsets[0], window) >= edge_distance(n.onset, offsets[1], window):
            out.append(n)
    out += [m for j, m in enumerate(second) if j not in taken and edge_distance(m.onset, offsets[1], window) > edge_distance(m.onset, offsets[0], window)]
    return sorted(out, key=lambda n: (n.onset, n.pitch))


def main():
    args = sys.argv[1:]
    passes = None
    if args[:1] == ["--shift"]:
        passes, args = [float(args[1])], args[2:]
    elif args[:1] == ["--single"]:
        passes, args = [0.0], args[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit("usage: run_yourmt3.py [--single | --shift SECONDS] <recording> <out.mid> [<recording> <out.mid> ...]")
    jobs = [(Path(a).resolve(), Path(b).resolve()) for a, b in zip(args[0::2], args[1::2])]
    os.chdir(TOOL)  # the model code finds its checkpoint relative to this folder
    sys.path.insert(0, str(TOOL / "amt" / "src"))
    sys.path.insert(0, str(TOOL))
    from dataclasses import replace

    import torchaudio
    from model_helper import load_model_checkpoint
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    from utils.utils import write_model_output_as_midi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_checkpoint(args=ARGS + ["-pr", "16" if device == "cuda" else "32"], device="cpu").to(device)
    frames = model.audio_cfg["input_frames"]
    window = frames / model.audio_cfg["sample_rate"]

    def transcribe(mono, sr, shift):
        if shift:
            mono = np.concatenate([np.zeros(int(round(shift * sr)), np.float32), mono])
        audio = torchaudio.functional.resample(torch.from_numpy(mono).unsqueeze(0), sr, model.audio_cfg["sample_rate"])
        segments = slice_padded_array(audio.numpy(), frames, frames)
        segments = torch.from_numpy(segments.astype("float32")).to(device).unsqueeze(1)
        with torch.no_grad():
            tokens, _ = model.inference_file(bsz=4, audio_segments=segments)
        starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
        notes_by_channel, errors = [], Counter()
        for ch in range(model.task_manager.num_decoding_channels):
            zipped, _, _ = model.task_manager.detokenize_list_batches([arr[:, ch, :] for arr in tokens], starts, return_events=True)
            notes, err = merge_zipped_note_events_and_ties_to_notes(zipped)
            notes_by_channel.append(notes)
            errors += err
        notes = mix_notes(notes_by_channel)
        if shift:  # take the added silence off again
            notes = [replace(n, onset=max(n.onset - shift, 0.0), offset=n.offset - shift) for n in notes if n.offset - shift > 0.0]
        return notes, errors

    for audio_path, out_path in jobs:
        y, sr = sf.read(str(audio_path), always_2d=True)
        mono = y.mean(axis=1).astype("float32")
        if passes is None:
            first, errors = transcribe(mono, sr, 0.0)
            second, _ = transcribe(mono, sr, window / 2)
            notes = stitch(first, second, window, window / 2)
            how = "two passes joined (%d and %d notes -> %d)" % (len(first), len(second), len(notes))
        else:
            notes, errors = transcribe(mono, sr, passes[0])
            how = "one pass" + (" with windows moved by %.3f s" % passes[0] if passes[0] else "")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:  # the model writes into a folder of its own
            write_model_output_as_midi(notes, tmp + "/", out_path.stem, model.midi_output_inverse_vocab)
            os.replace(Path(tmp) / "model_output" / (out_path.stem + ".mid"), out_path)
        print("wrote", out_path, "|", how, "| decoding errors in the first pass", dict(errors), flush=True)


if __name__ == "__main__":
    main()
