import os
import hashlib
import re
import threading
import time
import uuid
from collections import OrderedDict

import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"

import numpy as np
from flask import (
    Flask, Response, jsonify, render_template, request,
    send_from_directory, stream_with_context,
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ---- Load the model once at startup -----------------------------------------
t0 = time.time()
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
print(f"Loading {MODEL_NAME} ...", flush=True)
tts = TTS(MODEL_NAME).to("cpu")
SPEAKERS = sorted(list(tts.synthesizer.tts_model.speaker_manager.speakers.keys()))
LANGUAGES = list(tts.synthesizer.tts_model.config.languages)
print(f"Model loaded in {time.time() - t0:.1f}s. Speakers: {len(SPEAKERS)}", flush=True)

# XTTS on CPU is not re-entrant -> serialize synthesis calls.
_lock = threading.Lock()

# Conditioning-latent cache (Auralis technique): compute gpt_cond_latent +
# speaker_embedding once per speaker/reference and reuse. Bit-identical audio.
_cond_cache: dict = {}
_cond_lock = threading.Lock()


def _get_latents(speaker=None, speaker_wav=None):
    """Return (gpt_cond_latent, speaker_embedding), memoized per voice."""
    m = tts.synthesizer.tts_model
    if speaker and speaker in m.speaker_manager.speakers:
        key = ("spk", speaker)
        with _cond_lock:
            if key not in _cond_cache:
                _cond_cache[key] = tuple(m.speaker_manager.speakers[speaker].values())
            return _cond_cache[key]
    # reference-audio clone -> key by file content hash
    h = hashlib.sha256()
    with open(speaker_wav, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    key = ("ref", h.hexdigest())
    with _cond_lock:
        if key not in _cond_cache:
            v = m.clone_voice(speaker_wav)
            _cond_cache[key] = (v["gpt_conditioning_latents"], v["speaker_embedding"])
        return _cond_cache[key]


# ---- Semantic / audio cache (Future AGI pattern) ---------------------------
# Cache rendered audio keyed on NORMALIZED phrase + voice + language so that
# greetings, holds, closings and any repeated phrase ("earlier scene response")
# return instantly instead of re-synthesizing. On CPU XTTS (~7s / sentence) a
# hit saves the ENTIRE generation, not just the 120-350ms a GPU TTS provider
# would save -- a much bigger win.

def normalize_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _ref_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _voice_id(mode, speaker, ref_path=None):
    if mode == "clone" and ref_path:
        return "clone:" + _ref_hash(ref_path)
    return f"builtin:{speaker}"


def audio_cache_key(text, voice_id, language) -> str:
    # phrase | voice | language | sample-rate | channels  (per blog: always
    # include voice + language so different voices/locales never collide)
    raw = f"{normalize_phrase(text)}|{voice_id}|{language}|24000|mono"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_wav_path(key: str) -> str:
    return os.path.join(app.config["OUTPUT_FOLDER"], f"cache_{key[:24]}.wav")


class AudioCache:
    """LRU map: cache-key -> stable wav path under OUTPUT_FOLDER."""

    def __init__(self, maxsize=512):
        self._lock = threading.Lock()
        self._paths: "OrderedDict[str, str]" = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            p = self._paths.get(key)
            if p and os.path.exists(p):
                self._paths.move_to_end(key)
                self.hits += 1
                return p
            if p:
                del self._paths[key]  # stale file
            self.misses += 1
            return None

    def put(self, key, path):
        with self._lock:
            self._paths[key] = path
            self._paths.move_to_end(key)
            while len(self._paths) > self.maxsize:
                _, old = self._paths.popitem(last=False)
                try:
                    os.remove(old)
                except OSError:
                    pass

    def stats(self):
        with self._lock:
            n = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / n, 3) if n else 0.0,
                "entries": len(self._paths),
                "maxsize": self.maxsize,
            }


_audio_cache = AudioCache()


def _pcm_from_wav(path):
    """Load a cached wav as int16 mono PCM bytes (for the /stream fast path)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)


@app.route("/stream", methods=["POST"])
def stream():
    """Real-time streaming TTS (Baseten / inference_stream pattern).

    Streams raw 16-bit PCM (24 kHz mono) chunk-by-chunk as XTTS generates it,
    so the browser can start playback almost immediately and keep playing while
    the rest of the sentence is still being synthesized.
    """
    text = (request.form.get("text") or "").strip()
    language = request.form.get("language", "en")
    mode = request.form.get("mode", "builtin")
    speaker = request.form.get("speaker", SPEAKERS[0] if SPEAKERS else None)
    chunk_size = int(request.form.get("chunk_size", 4))  # GPT tokens per chunk (lower = faster first audio)

    if not text:
        return jsonify({"error": "Text is required."}), 400
    if language not in LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'."}), 400

    # Resolve reference path (clone) so we can build the cache key early.
    ref_path = None
    if mode == "clone":
        f = request.files.get("reference")
        if f is None or f.filename == "":
            return jsonify({"error": "A reference audio file is required for cloning."}), 400
        safe = secure_filename(f.filename) or "ref.wav"
        ref_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex[:8]}_{safe}")
        f.save(ref_path)
    elif not speaker or speaker not in SPEAKERS:
        return jsonify({"error": "Choose a valid speaker."}), 400

    # ---- Cache lookup (skips conditioning + inference entirely on a hit) ----
    akey = audio_cache_key(text, _voice_id(mode, speaker, ref_path), language)
    cached = _audio_cache.get(akey)
    if cached:
        print(f"[stream] cache HIT -> {os.path.basename(cached)}", flush=True)
        t0 = time.time()

        def serve_cached():
            pcm = _pcm_from_wav(cached)
            b = 8192  # large chunks: we already have the audio, ship it fast
            for i in range(0, len(pcm), b):
                yield pcm[i:i + b].tobytes()
            print(
                f"[stream] cache-hit served {(time.time()-t0)*1000:.0f} ms",
                flush=True,
            )

        return Response(
            stream_with_context(serve_cached()),
            mimetype="audio/pcm",
            direct_passthrough=True,
            headers={"X-Accel-Buffering": "no", "X-Cache": "HIT"},
        )

    # ---- Cache miss: resolve conditioning, then stream-generate ------------
    try:
        if mode == "clone":
            gcl, se = _get_latents(speaker_wav=ref_path)
        else:
            gcl, se = _get_latents(speaker=speaker)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    m = tts.synthesizer.tts_model
    t_start = time.time()
    first_sent = {"done": False}
    cachepath = _cache_wav_path(akey)
    chunks = []

    def generate():
        # Hold the lock for the duration of the stream (XTTS CPU is not re-entrant).
        with _lock:
            streamer = m.inference_stream(
                text,
                language,
                gcl,
                se,
                stream_chunk_size=chunk_size,
                enable_text_splitting=True,
            )
            for wav_chunk in streamer:
                arr = wav_chunk.detach().cpu().numpy()
                chunks.append(arr)
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                if not first_sent["done"]:
                    first_sent["done"] = True
                    print(
                        f"[stream] time-to-first-chunk: {(time.time()-t_start)*1000:.0f} ms",
                        flush=True,
                    )
                yield pcm.tobytes()
        # Persist the finished audio so the next identical phrase is a cache hit.
        try:
            if chunks:
                sf.write(cachepath, np.concatenate(chunks), 24000)
                _audio_cache.put(akey, cachepath)
        except Exception as e:  # pragma: no cover
            print(f"[stream] cache store failed: {e}", flush=True)

    headers = {"X-Accel-Buffering": "no", "X-Cache": "MISS"}  # disable proxy buffering
    return Response(
        stream_with_context(generate()),
        mimetype="audio/pcm",
        direct_passthrough=True,
        headers=headers,
    )

# Language labels for the UI
LANGUAGE_LABELS = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh-cn": "Chinese", "hu": "Hungarian", "ko": "Korean", "ja": "Japanese",
    "hi": "Hindi",
}


@app.route("/")
def index():
    return render_template("index.html", speakers=SPEAKERS, languages=LANGUAGES,
                           labels=LANGUAGE_LABELS)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME,
                    "speakers": len(SPEAKERS), "languages": LANGUAGES})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = (request.form.get("text") or "").strip()
    language = request.form.get("language", "en")
    mode = request.form.get("mode", "builtin")  # 'builtin' or 'clone'
    speaker = request.form.get("speaker", SPEAKERS[0] if SPEAKERS else None)

    if not text:
        return jsonify({"error": "Text is required."}), 400
    if language not in LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'."}), 400

    kwargs = {"text": text, "language": language}
    fname = f"out_{uuid.uuid4().hex[:12]}.wav"
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], fname)

    if mode == "clone":
        f = request.files.get("reference")
        if f is None or f.filename == "":
            return jsonify({"error": "A reference audio file is required for cloning."}), 400
        safe = secure_filename(f.filename) or "ref.wav"
        ref_path = os.path.join(app.config["UPLOAD_FOLDER"],
                                f"{uuid.uuid4().hex[:8]}_{safe}")
        f.save(ref_path)
        kwargs["speaker_wav"] = ref_path
    else:
        if not speaker or speaker not in SPEAKERS:
            return jsonify({"error": "Choose a valid speaker."}), 400
        kwargs["speaker"] = speaker

    t0 = time.time()
    try:
        with _lock:
            tts.tts_to_file(file_path=out_path, **kwargs)
    except Exception as e:  # pragma: no cover - surface errors to the UI
        return jsonify({"error": str(e)}), 500
    elapsed = time.time() - t0

    return jsonify({
        "audio_url": f"/audio/{fname}",
        "seconds": round(elapsed, 2),
        "mode": mode,
        "language": language,
        "speaker": speaker if mode != "clone" else None,
    })


@app.route("/audio/<path:fname>")
def audio(fname):
    return send_from_directory(app.config["OUTPUT_FOLDER"], fname, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False, threaded=True)
