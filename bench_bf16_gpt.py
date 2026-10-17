import os, time, timeit
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))
SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"
lang="hi"
v = m.clone_voice(ref); gcl, se = v["gpt_conditioning_latents"], v["speaker_embedding"]
gcl=gcl.to(m.device); se=se.to(m.device)
tok = torch.IntTensor(m.tokenizer.encode(SENT.strip().lower(), lang=lang)).unsqueeze(0)

def gen():
    with torch.no_grad():
        return m.gpt.generate(cond_latents=gcl, text_inputs=tok, do_sample=True,
            top_p=0.85, top_k=50, temperature=0.75, num_return_sequences=m.gpt_batch_size,
            num_beams=1, length_penalty=1.0, repetition_penalty=10.0, output_attentions=False)

_ = gen()
base = timeit.timeit(gen, number=1)
print(f"[baseline fp32 gpt.generate] : {base*1000:.0f} ms")

# 1) bf16 autocast scoped to ONLY gpt.generate
def gen_bf16():
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        return m.gpt.generate(cond_latents=gcl, text_inputs=tok, do_sample=True,
            top_p=0.85, top_k=50, temperature=0.75, num_return_sequences=m.gpt_batch_size,
            num_beams=1, length_penalty=1.0, repetition_penalty=10.0, output_attentions=False)
try:
    _ = gen_bf16()
    t=timeit.timeit(gen_bf16, number=1)
    print(f"[bf16 autocast gpt only]    : {t*1000:.0f} ms  ({base/t:.2f}x)")
except Exception as e:
    print(f"[bf16 autocast gpt only] FAILED: {repr(e)[:200]}")

# 2) fully cast gpt to bf16 + bf16 inputs
try:
    m.gpt.to(torch.bfloat16)
    gcl_b = gcl.to(torch.bfloat16); tok_b = tok
    def gen_bf16_cast():
        with torch.no_grad():
            return m.gpt.generate(cond_latents=gcl_b, text_inputs=tok_b, do_sample=True,
                top_p=0.85, top_k=50, temperature=0.75, num_return_sequences=m.gpt_batch_size,
                num_beams=1, length_penalty=1.0, repetition_penalty=10.0, output_attentions=False)
    _ = gen_bf16_cast()
    t=timeit.timeit(gen_bf16_cast, number=1)
    print(f"[gpt fully cast bf16]        : {t*1000:.0f} ms  ({base/t:.2f}x)")
    m.gpt.to(torch.float32)  # restore
except Exception as e:
    print(f"[gpt fully cast bf16] FAILED: {repr(e)[:200]}")
    m.gpt.to(torch.float32)
