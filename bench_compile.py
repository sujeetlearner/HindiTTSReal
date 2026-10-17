import os, time, timeit
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))
SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"
v = m.clone_voice(ref); gcl, se = v["gpt_conditioning_latents"], v["speaker_embedding"]

def run():
    return m.inference(SENT, "hi", gcl, se)

_ = run()
base = timeit.timeit(run, number=1)
print(f"[baseline]           : {base*1000:.0f} ms")

# compile the GPT decoder forward
try:
    print("compiling m.gpt ...", flush=True)
    m.gpt = torch.compile(m.gpt, mode="default", dynamic=False)
    _ = run()  # triggers compile (slow first time)
    t1 = timeit.timeit(run, number=1)
    print(f"[compiled gpt]       : {t1*1000:.0f} ms  ({base/t1:.2f}x)")
except Exception as e:
    print("compile gpt FAILED:", repr(e)[:300])

# also try compiling the whole xtts model
try:
    m2 = tts.synthesizer.tts_model
    m2 = torch.compile(m2, mode="default")
    tts.synthesizer.tts_model = m2
    v2 = m2.clone_voice(ref)
    def run2():
        return m2.inference(SENT, "hi", v2["gpt_conditioning_latents"], v2["speaker_embedding"])
    _ = run2()
    t2 = timeit.timeit(run2, number=1)
    print(f"[compiled full model] : {t2*1000:.0f} ms  ({base/t2:.2f}x)")
except Exception as e:
    print("compile full FAILED:", repr(e)[:300])
