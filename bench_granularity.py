"""Cache-granularity benchmark: how much time/cost does caching save at each
text segmentation level, and what does it cost in audio quality?

Renders the SAME scene text under 5 configurations:

    none     no cache (baseline -- full generation every time)
    whole    cache the entire text as one unit (exact-repeat only)
    sentence cache each sentence  (XTTS already generates per-sentence -> identical audio)
    phrase   cache each clause    (split at punctuation -> seams at pauses)
    word     cache each word      (each word generated standalone -> choppy)

For each we measure:
  * cold   : first render (cache empty)
  * warm   : identical text again (full-repeat cache benefit)
  * partial: scene with ONE sentence changed after the scene is cached
  * calls  : number of TTS model invocations (cost / compute proxy)

and write a wav per granularity so you can HEAR the quality difference.

Run:  venv/bin/python bench_granularity.py
"""
import os
import re
import time

os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import soundfile as sf
from TTS.api import TTS

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER = "Claribel Dervla"
LANGUAGE = "en"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# inter-segment silence, tuned per granularity (coqui uses 10000 between sentences)
PAD = {"none": 10000, "whole": 10000, "sentence": 10000, "phrase": 3000, "word": 1000}

# --------------------------------------------------------------------------- #
# A "scene": an NPC greeting with several lines that repeat across turns.
# --------------------------------------------------------------------------- #
SCENE = (
    "Welcome traveler! The old gate is just ahead. "
    "Be careful, the road is dangerous at night. "
    "Welcome traveler! Have you come seeking the crystal? "
    "Be careful, the road is dangerous at night. "
    "The old gate is just ahead, follow the lanterns."
)

# Turn 2: identical except the final sentence (a scene that mostly repeats).
SCENE_TURN2 = (
    "Welcome traveler! The old gate is just ahead. "
    "Be careful, the road is dangerous at night. "
    "Welcome traveler! Have you come seeking the crystal? "
    "Be careful, the road is dangerous at night. "
    "The old gate is closed, turn back now."
)

GRANULARITIES = ["none", "sentence", "phrase", "word"]


def split_sentences(tts, text):
    try:
        s = tts.synthesizer.split_into_sentences(text)
    except Exception:
        s = [text]
    return [x for x in s if x.strip()] or [text]


def split_phrases(tts, text):
    out = []
    for sent in split_sentences(tts, text):
        # split at clause punctuation (pause boundaries), keep the punctuation
        parts = re.split(r"(?<=[,;:])\s+", sent)
        parts = [p.strip() for p in parts if p and p.strip()]
        out.extend(parts or [sent])
    return out


def split_words(text):
    return [w for w in text.split() if w]


def segment(tts, text, gran):
    if gran == "whole":
        return [text]
    if gran == "sentence":
        return split_sentences(tts, text)
    if gran == "phrase":
        return split_phrases(tts, text)
    if gran == "word":
        return split_words(text)
    # "none" -> sentence units but never cached (baseline)
    return split_sentences(tts, text)


def seg_key(seg, speaker, language):
    norm = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", seg.lower().strip()))
    raw = f"{norm}|{speaker}|{language}|24000|mono"
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()


def render(tts, text, gran, cache, use_cache=True):
    """Return (elapsed_seconds, wav_float32, calls, hits, misses).

    ALL granularities (including the 'none' baseline) synthesize per-segment
    via tts.tts(seg, split_sentences=False). This avoids a single huge tts()
    call (which can thrash memory on CPU) and makes 'none' a fair baseline:
    same sentence units as 'sentence' granularity, just with caching disabled.
    """
    t0 = time.time()
    calls = hits = misses = 0

    if gran == "none":
        segs = split_sentences(tts, text)
        use_cache = False
    else:
        segs = segment(tts, text, gran)

    pad = PAD[gran]
    pieces = []
    for i, seg in enumerate(segs):
        key = seg_key(seg, SPEAKER, LANGUAGE)
        if use_cache and key in cache:
            pieces.append(cache[key])
            hits += 1
        else:
            w = np.asarray(
                tts.tts(seg, speaker=SPEAKER, language=LANGUAGE, split_sentences=False),
                dtype=np.float32,
            )
            if w.ndim > 1:
                w = w.mean(axis=1)
            if use_cache:
                cache[key] = w
            pieces.append(w)
            misses += 1
            calls += 1
        if i < len(segs) - 1:
            pieces.append(np.zeros(pad, dtype=np.float32))
    wav = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return time.time() - t0, wav, calls, hits, misses


def main():
    print(f"Loading {MODEL} ...", flush=True)
    t_load = time.time()
    tts = TTS(MODEL).to("cpu")
    print(f"Model loaded in {time.time() - t_load:.1f}s\n", flush=True)

    n_sent = len(split_sentences(tts, SCENE))
    n_word = len(split_words(SCENE))
    print(f"Scene: {n_sent} sentences, {n_word} words\n")
    print("=" * 96)

    rows = []
    quality_wavs = {}

    for gran in GRANULARITIES:
        print(f"\n--- granularity: {gran} ---", flush=True)
        cache = {}
        use_cache = gran != "none"

        try:
            # COLD (empty cache)
            cold, wav_cold, calls_cold, _, miss_cold = render(tts, SCENE, gran, cache, use_cache)
            if use_cache:
                # WARM (exact same scene again)
                warm, wav_warm, calls_warm, hits_warm, miss_warm = render(tts, SCENE, gran, cache, use_cache)
                # PARTIAL (scene turn 2, one sentence changed)
                part, wav_part, calls_part, hits_part, miss_part = render(tts, SCENE_TURN2, gran, cache, use_cache)
            else:
                # No cache: every render is a full regeneration. Skip the repeats.
                warm = part = cold
                calls_warm = calls_part = calls_cold
                hits_warm = hits_part = 0
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            rows.append((gran, float("nan"), float("nan"), float("nan"), str(e), ""))
            continue

        speedup = (cold / warm) if warm > 1e-6 else float("inf")
        saved_pct = (1 - warm / cold) * 100 if cold > 0 else 0

        fname = f"gran_{gran}.wav"
        sf.write(os.path.join(OUT_DIR, fname), wav_cold, 24000)
        quality_wavs[gran] = fname

        print(f"  cold    : {cold:6.2f}s  ({calls_cold} TTS calls)")
        print(f"  warm    : {warm:6.2f}s  ({calls_warm} TTS calls, {hits_warm} cache hits)")
        print(f"  partial : {part:6.2f}s  ({calls_part} new TTS calls, {hits_part} hits, 1 sentence changed)")
        print(f"  speedup : {speedup:6.1f}x   saved {saved_pct:.1f}% on full repeat")
        if calls_cold:
            print(f"  per-call: {cold / calls_cold * 1000:.0f} ms average")

        rows.append((gran, cold, warm, part, speedup, calls_warm, fname))

    print("\n" + "=" * 96)
    print("SUMMARY (scene with repeated lines)")
    print("=" * 96)
    hdr = f"{'granularity':<12}{'cold(s)':>9}{'warm(s)':>9}{'partial(s)':>11}{'speedup':>9}{'warm calls':>11}{'audio':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if len(r) == 6:  # failure row
            print(f"{r[0]:<12}{'FAILED':>9}  {r[5]}")
            continue
        gran, cold, warm, part, speedup, calls_warm, fname = r
        sp = f"{speedup:.1f}x" if speedup != float("inf") else "1.0x"
        print(f"{gran:<12}{cold:>9.2f}{warm:>9.2f}{part:>11.2f}{sp:>9}{calls_warm:>11}{('outputs/' + fname):>16}")

    print("\nListen to the quality samples (same text, different granularity):")
    for gran, fname in quality_wavs.items():
        print(f"  {gran:<10} -> webapp/outputs/{fname}")

    print("""
How to read this:
  * 'warm' near 0  => cache made a full repeat ~instant (the big win).
  * 'warm calls'=0  => nothing regenerated (zero compute cost on repeat).
  * sentence speedup with IDENTICAL quality is the safe win.
  * phrase/word may show MORE speedup, but listen to the samples:
    word-level audio is choppy (each word generated in isolation, no
    coarticulation/prosody across the sentence). Trade speed for quality.
""")


if __name__ == "__main__":
    main()
