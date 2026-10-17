import os, time, json, statistics
os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(MODEL).to("cpu")
model = tts.synthesizer.tts_model
m = model

text = ("नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है। "
        "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है और यह काफी प्राकृतिक लगती है।")
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = None
for f in sorted(os.listdir(REF)):
    if f.lower().endswith((".wav", ".mp3", ".webm")):
        ref = os.path.join(REF, f); break
print("ref:", ref)

SENTS = ["नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।",
         "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है और यह काफी प्राकृतिक लगती है।"]

# --- 1. conditioning (clone_voice) cost ---
t0=time.time(); voice = m.clone_voice(ref); t_cond=time.time()-t0
gcl, se = voice["gpt_conditioning_latents"], voice["speaker_embedding"]
print(f"clone_voice (conditioning) : {t_cond*1000:.0f} ms")

# --- 2. inference (GPT decode) cost, per sentence, with cached latents ---
inf_times=[]
for s in SENTS:
    t0=time.time(); _ = m.inference(s, "hi", gcl, se); inf_times.append(time.time()-t0)
print(f"inference (GPT+decode)     : {np.mean(inf_times)*1000:.0f} ms/sentence "
      f"(min {min(inf_times)*1000:.0f}, max {max(inf_times)*1000:.0f})")

# --- 3. full baseline synthesize() path (recomputes clone_voice per sentence) ---
t0=time.time()
for s in SENTS:
    _ = m.synthesize(text=s, speaker_wav=ref, language="hi")
t_base=time.time()-t0
print(f"BASELINE synthesize x{len(SENTS)} (recompute cond): {t_base*1000:.0f} ms")

# --- 4. optimized: condition once, inference per sentence ---
t0=time.time()
v = m.clone_voice(ref)
for s in SENTS:
    _ = m.inference(s, "hi", v["gpt_conditioning_latents"], v["speaker_embedding"])
t_opt=time.time()-t0
print(f"OPTIMIZED cond-once + infer x{len(SENTS)}: {t_opt*1000:.0f} ms")

print(f"\nSpeedup (cond cached): {t_base/t_opt:.2f}x  "
      f"(baseline {t_base*1000:.0f}ms -> {t_opt*1000:.0f}ms)")
