"""Find beats and the first beat of each bar with the Beat This! model.

Beat This! (Foscarin, Schlüter and Widmer, ISMIR 2024; MIT licence) is a beat and downbeat
tracker trained across many styles. This wrapper expects its code under tools/beat_this/ (the
authors' GitHub repository) and runs in the environment that already has PyTorch (.venv_mt3).
The model's weights are downloaded by its own code on first use and cached by PyTorch.

Writes <out.json> with "beats" and "downbeats" in seconds.

usage: .venv_mt3/bin/python scripts/run_beats.py <recording> <out.json> [<recording> <out.json> ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "beat_this"))


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit("usage: run_beats.py <recording> <out.json> [<recording> <out.json> ...]")
    import torch
    from beat_this.inference import File2Beats

    track = File2Beats(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)
    for audio, out in zip(args[0::2], args[1::2]):
        beats, downbeats = track(audio)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": "Beat This! final0", "beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(d), 4) for d in downbeats]}))
        print("wrote", out, "| beats", len(beats), "| downbeats", len(downbeats), flush=True)


if __name__ == "__main__":
    main()
