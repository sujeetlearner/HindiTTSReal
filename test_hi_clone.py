import os, time
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

# Voice cloning: give a REFERENCE AUDIO (6-30s of clean speech).
# For natural HINDI, use a clean recording of a Hindi speaker as speaker_wav.
# Here we use the bundled English sample just to demonstrate the API works.
# Replace "ref/hindi_reference.wav" with your own Hindi audio for natural Hindi.

text_hi = (
    "नमस्ते! यह आपकी आवाज़ पर आधारित एक प्राकृतिक हिंदी आवाज़ है। "
    "मैं आपके संदेश को स्पष्ट और स्वाभाविक ढंग से पढ़ती हूँ।"
)

t0 = time.time()
tts.tts_to_file(
    text=text_hi,
    speaker_wav="ref/en_sample.wav",   # <-- put a clean Hindi recording here
    language="hi",
    file_path="out_hi_cloned.wav",
)
print(f"Cloned HI synthesis: {time.time()-t0:.1f}s -> out_hi_cloned.wav", flush=True)
