import os, time, torch
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model
SPK = "Claribel Dervla"
gcl, se = m.speaker_manager.speakers[SPK].values()
TEXT = "Hello! Streaming test."
print("code_stride_len:", m.gpt.code_stride_len, flush=True)

def measure(cs):
    t0 = time.time(); first = None; samples = 0; n = 0
    for w in m.inference_stream(TEXT, "en", gcl, se, stream_chunk_size=cs, enable_text_splitting=True):
        if first is None: first = time.time() - t0
        samples += len(w); n += 1
    return first, time.time()-t0, n, samples

print(f"{'chunk':>6} {'TTFB(s)':>9} {'total(s)':>9} {'chunks':>7} {'audio(s)':>9}", flush=True)
for cs in [1, 4, 20]:
    first, total, n, samples = measure(cs)
    print(f"{cs:>6} {first:>9.2f} {total:>9.2f} {n:>7} {samples/24000:>9.2f}", flush=True)
