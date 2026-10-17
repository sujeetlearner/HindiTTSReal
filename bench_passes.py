import os, time
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
m = tts.synthesizer.tts_model
REF = "/Users/sujeetkumarvishwakarma/coqui-tts-demo/webapp/uploads"
ref = next(os.path.join(REF,f) for f in sorted(os.listdir(REF)) if f.lower().endswith(".wav"))
SENT = "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है।"
lang = "hi"
v = m.clone_voice(ref); gcl, se = v["gpt_conditioning_latents"], v["speaker_embedding"]

text = [SENT.strip().lower()]
text_tokens = torch.IntTensor(m.tokenizer.encode(text[0], lang=lang)).unsqueeze(0)
gcl = gcl.to(m.device); se = se.to(m.device)

with torch.no_grad():
    # pass 1: generate
    t0=time.time()
    gpt_codes = m.gpt.generate(cond_latents=gcl, text_inputs=text_tokens, do_sample=True,
        top_p=0.85, top_k=50, temperature=0.75, num_return_sequences=m.gpt_batch_size,
        num_beams=1, length_penalty=1.0, repetition_penalty=10.0, output_attentions=False)
    t_gen=time.time()-t0
    ntok = gpt_codes.shape[-1]

    expected_output_len = torch.tensor([gpt_codes.shape[-1]*m.gpt.code_stride_len], device=text_tokens.device)
    text_len = torch.tensor([text_tokens.shape[-1]], device=m.device)
    # pass 2: latent extraction
    t0=time.time()
    gpt_latents = m.gpt(text_tokens, text_len, gpt_codes, expected_output_len,
                        cond_latents=gcl, return_attentions=False, return_latent=True)
    t_lat=time.time()-t0
    # vocoder
    t0=time.time()
    wav = m.hifigan_decoder(gpt_latents, g=se)
    t_voc=time.time()-t0

print(f"audio tokens generated : {ntok}")
print(f"pass1 gpt.generate (AR): {t_gen*1000:.0f} ms")
print(f"pass2 gpt latent fwd   : {t_lat*1000:.0f} ms")
print(f"hifigan vocoder        : {t_voc*1000:.0f} ms")
tot=t_gen+t_lat+t_voc
print(f"---\nmodel total           : {tot*1000:.0f} ms")
print(f"pass2 share            : {t_lat/tot*100:.1f}%  -> eliminating it saves ~{t_lat*1000:.0f}ms")
print(f"vocoder share          : {t_voc/tot*100:.1f}%")
