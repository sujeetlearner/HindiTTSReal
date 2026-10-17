import os, sys, time, torch
os.environ["COQUI_TOS_AGREED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from TTS.api import TTS
from fast_engine import FastXTTS

torch.set_num_threads(6)  # match stock default for a fair algorithmic comparison
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
fast = FastXTTS(tts)
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))

TEXT3 = ("नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है। "
         "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है। "
         "यह काफी प्राकृतिक लगती है।")
SENT1 = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"

def baseline(text):
    t0=time.time(); tts.tts_to_file(text=text, language="hi", speaker_wav=ref, file_path="/tmp/c_bl.wav"); return time.time()-t0
def fastp(text, clear=False):
    if clear: fast.cache._store.clear()
    t0=time.time(); fast.synthesize(text, "hi", speaker_wav=ref); return time.time()-t0

# warm both
baseline("warm up one."); fastp("warm up two.")

def minof(fn, text, n=3, **kw):
    return min(fn(text, **kw) for _ in range(n))

print("=== 1 sentence (min of 3, threads=6, matched) ===")
b=minof(baseline,SENT1); f=minof(fastp,SENT1)
print(f"  baseline {b*1000:.0f}ms | fast {f*1000:.0f}ms | {b/f:.2f}x")

print("=== 3 sentences (min of 3) ===")
b=minof(baseline,TEXT3); f=minof(fastp,TEXT3)
print(f"  baseline {b*1000:.0f}ms | fast {f*1000:.0f}ms | {b/f:.2f}x")

print("=== 3 sentences, repeated clone (conditioning cache) ===")
fc=minof(fastp,TEXT3,clear=True)   # fresh cache each time (worst case)
fw=minof(fastp,TEXT3,clear=False)  # warm cache
print(f"  fast cold-cache {fc*1000:.0f}ms | fast warm-cache {fw*1000:.0f}ms | cache helps {(fc-fw)*1000:.0f}ms")
