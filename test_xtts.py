import os, time
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

print("loading XTTS v2 ...", flush=True)
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
print("built-in speakers:", speakers[:8], flush=True)
SPEAKER = speakers[0]  # e.g. 'Claribel Dervla'
print("using speaker:", SPEAKER, flush=True)
print("supported languages:", tts.synthesizer.tts_model.config.languages, flush=True)

text_en = (
    "Hello! This is a quality test of the Coqui text to speech toolkit. "
    "It is battle tested in research and production, and it sounds quite natural."
)
t0 = time.time()
tts.tts_to_file(text=text_en, speaker=SPEAKER, language="en", file_path="out_en_xtts.wav")
print(f"EN synthesis: {time.time()-t0:.1f}s -> out_en_xtts.wav", flush=True)

text_hi = (
    "नमस्ते! यह कोक्वी टेक्स्ट टू स्पीच टूलकिट की गुणवत्ता परीक्षण है। "
    "यह अनुसंधान और उत्पादन दोनों में प्रमाणित है और यह काफी प्राकृतिक लगती है।"
)
t0 = time.time()
tts.tts_to_file(text=text_hi, speaker=SPEAKER, language="hi", file_path="out_hi_xtts.wav")
print(f"HI synthesis: {time.time()-t0:.1f}s -> out_hi_xtts.wav", flush=True)
