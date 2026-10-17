import os, time, sys
os.environ["COQUI_TOS_AGREED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from TTS.api import TTS
from fast_engine import FastXTTS, configure_threads

print("threads pinned to:", configure_threads(), flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
fast = FastXTTS(tts)

REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))

TEXT_SHORT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"
TEXT_LONG = ("नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है। "
             "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है और यह काफी प्राकृतिक लगती है। "
             "आप इसे अपनी आवाज़ पर आधारित बना सकते हैं।")

def baseline(text):
    t0=time.time()
    tts.tts_to_file(text=text, language="hi", speaker_wav=ref, file_path="/tmp/b_bl.wav")
    return time.time()-t0

def fastpath(text, fresh_cache=False):
    if fresh_cache: fast.cache._store.clear()
    t0=time.time()
    wav,_ = fast.synthesize(text, "hi", speaker_wav=ref)
    return time.time()-t0, wav

# warm
baseline("warmup sentence one."); fastpath("warmup sentence two.")

print("\n--- SHORT (1 sentence) ---")
tb=baseline(TEXT_SHORT); tf,_=fastpath(TEXT_SHORT, fresh_cache=True)
print(f"baseline {tb*1000:.0f}ms | fast {tf*1000:.0f}ms | {tb/tf:.2f}x")

print("\n--- LONG (3 sentences) ---")
tb=baseline(TEXT_LONG); tf,_=fastpath(TEXT_LONG, fresh_cache=True)
print(f"baseline {tb*1000:.0f}ms | fast {tf*1000:.0f}ms | {tb/tf:.2f}x")

print("\n--- REPEAT request (cached conditioning) ---")
tf1,_=fastpath(TEXT_SHORT, fresh_cache=True)   # cold cache
tf2,_=fastpath(TEXT_SHORT, fresh_cache=False)  # warm cache
print(f"fast cold {tf1*1000:.0f}ms | fast warm {tf2*1000:.0f}ms | cache saves {(tf1-tf2)*1000:.0f}ms")
