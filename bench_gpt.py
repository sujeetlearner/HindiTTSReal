import os, time, timeit
os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
import torch
from TTS.api import TTS

print("torch", torch.__version__)
print("threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
print("mkldnn", torch.backends.mkldnn.is_available(), torch.backends.mkldnn.enabled)
print("cpu capability:", torch.backends.cpu.get_cpu_capability())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model

REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = None
for f in sorted(os.listdir(REF)):
    if f.lower().endswith((".wav", ".mp3", ".webm")):
        ref = os.path.join(REF, f); break

SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"

# precompute latents once
v = m.clone_voice(ref)
gcl, se = v["gpt_conditioning_latents"], v["speaker_embedding"]

def run_infer():
    return m.inference(SENT, "hi", gcl, se)

# --- baseline fp32 (warm) ---
_ = run_infer()
t = timeit.timeit(run_infer, number=1)
print(f"\n[baseline fp32]           : {t*1000:.0f} ms")

# --- bf16 autocast ---
def run_bf16():
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        return m.inference(SENT, "hi", gcl, se)
try:
    _ = run_bf16()
    t_bf = timeit.timeit(run_bf16, number=1)
    print(f"[bf16 autocast]           : {t_bf*1000:.0f} ms  ({t/t_bf:.2f}x)")
except Exception as e:
    t_bf = None; print(f"[bf16 autocast] FAILED: {e}")

# --- thread sweep to check oversubscription ---
import itertools
best_t, best_n = t, torch.get_num_threads()
for n in [4, 6, 8, 10]:
    torch.set_num_threads(n)
    _ = run_infer()
    tt = timeit.timeit(run_infer, number=1)
    print(f"[threads={n}]             : {tt*1000:.0f} ms  ({t/tt:.2f}x)")
    if tt < best_t: best_t, best_n = tt, n
torch.set_num_threads(best_n)
print(f"best thread count: {best_n}")
