import os, time, sys
os.environ["COQUI_TOS_AGREED"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import torch
from TTS.api import TTS
from fast_engine import FastXTTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
fast = FastXTTS(tts)
m = fast.model
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))
SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"

gcl, se = fast._latents_for(speaker_wav=ref)   # cached

def stock_inference():
    with torch.no_grad():
        return m.inference(SENT, "hi", gcl, se)

def mine():
    lat, se_d = fast._gpt_latents(SENT, "hi", gcl, se)
    return fast._vocode(lat, se_d)

# thread sweep on STOCK inference
print("=== stock model.inference, thread sweep ===")
import timeit
for n in [4, 6, 8, 10]:
    torch.set_num_threads(n)
    _ = stock_inference()
    t = timeit.timeit(stock_inference, number=1)
    print(f"  threads={n}: {t*1000:.0f} ms")

# compare stock vs mine at best
torch.set_num_threads(10)
t_stock = timeit.timeit(stock_inference, number=1)
t_mine = timeit.timeit(mine, number=1)
print(f"\n=== threads=10 ===")
print(f"stock model.inference : {t_stock*1000:.0f} ms")
print(f"mine _gpt_latents+voc : {t_mine*1000:.0f} ms  ({t_mine/t_stock:.2f}x vs stock)")

# multi-sentence: baseline(tts_to_file path) vs stock loop with cached latents
TEXT3 = [SENT, "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है।", "यह काफी प्राकृतिक लगती है।"]
print(f"\n=== 3-sentence loop, threads=10 ===")
t0=time.time()
for s in TEXT3: _ = m.inference(s, "hi", gcl, se)
t_stock3=time.time()-t0
t0=time.time()
for s in TEXT3:
    lat, se_d = fast._gpt_latents(s, "hi", gcl, se); _ = fast._vocode(lat, se_d)
t_mine3=time.time()-t0
print(f"stock loop 3-sent : {t_stock3*1000:.0f} ms")
print(f"mine loop 3-sent  : {t_mine3*1000:.0f} ms")
