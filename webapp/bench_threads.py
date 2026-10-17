import os, sys, timeit, torch
os.environ["COQUI_TOS_AGREED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))
v = m.clone_voice(ref); gcl, se = v["gpt_conditioning_latents"], v["speaker_embedding"]
SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"
def run():
    with torch.no_grad(): return m.inference(SENT, "hi", gcl, se)
print("=== thread sweep (min of 2 runs) ===")
for n in [1,2,3,4,5,6,8]:
    torch.set_num_threads(n)
    _ = run()
    t = min(timeit.timeit(run, number=1) for _ in range(2))
    print(f"  threads={n}: {t*1000:.0f} ms")
