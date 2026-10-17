import time
from TTS.api import TTS

MODEL = "tts_models/en/ljspeech/vits"
text = (
    "Hello! This is a quick quality test of the Coqui text to speech toolkit. "
    "It is battle tested in research and production, and it sounds quite natural."
)

t0 = time.time()
tts = TTS(model_name=MODEL, progress_bar=False).to("cpu")
print(f"model loaded in {time.time() - t0:.1f}s")

t0 = time.time()
tts.tts_to_file(text=text, file_path="out_en_vits.wav")
print(f"synthesized in {time.time() - t0:.1f}s -> out_en_vits.wav")
