"""Auralis-inspired speedups for Coqui XTTS v2 on CPU — quality-preserving.

Techniques ported from Auralis (github.com/astramind-ai/Auralis) that do NOT
change the generated audio:

  1. Speaker-conditioning latent caching  -> compute gpt_cond_latent +
     speaker_embedding ONCE per reference instead of on every sentence/request
     (stock coqui recomputes `clone_voice` per sentence via synthesizer loop).
     Bit-identical math, pure memoization.

  2. Sentence prefetch                    -> split + tokenize text up front.

  3. Vocoder/GPT pipelining               -> run HiFi-GAN for sentence i in a
     worker thread while the GPT autoregressively decodes sentence i+1
     (Auralis runs the vocoder via asyncio.to_thread).

  4. Optimal CPU thread configuration     -> M2 has 10 cores; stock torch used 6.

  5. Per-sentence streaming (optional)    -> yield each sentence as soon as it is
     done, so time-to-first-audio is one sentence, not the whole text.

The dominant cost (AR GPT decode, ~87% of time) is unchanged — that is the stage
Auralis accelerates with vLLM, which requires an NVIDIA GPU and is not available
on Apple Silicon. These techniques trim the remaining ~13% + eliminate redundant
conditioning + improve latency.
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator

import numpy as np
import torch


def configure_threads(num_cores: int | None = None) -> int:
    """Pin intra-op threads to the physical core count (measured ~1.1x on M2)."""
    if num_cores is None:
        num_cores = os.cpu_count() or 4
    # Cap at 10 (M2 Pro perf+eff); going above measured *slower* on this machine.
    num_cores = min(num_cores, 10)
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(1, min(4, num_cores // 2)))
    return num_cores


class ConditioningCache:
    """Memoize XTTS conditioning latents keyed by reference-audio content.

    Reusing identical tensors is bit-identical to recomputing them, so audio
    quality is unchanged; we only skip redundant CPU work.
    """

    def __init__(self, maxsize: int = 16):
        self._lock = threading.Lock()
        self._store: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._maxsize = maxsize

    @staticmethod
    def _key_speaker(name: str) -> str:
        return "spk:" + name

    @staticmethod
    def _key_files(paths) -> str:
        h = hashlib.sha256()
        for p in paths if isinstance(paths, (list, tuple)) else [paths]:
            h.update(str(p).encode())
            try:
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            except OSError:
                h.update(b"unreadable")
        return "ref:" + h.hexdigest()

    def get_or_compute(self, key: str, compute_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]]):
        with self._lock:
            if key in self._store:
                return self._store[key]
        latents = compute_fn()
        with self._lock:
            if len(self._store) >= self._maxsize:
                self._store.pop(next(iter(self._store)))
            self._store[key] = latents
        return latents


class FastXTTS:
    """Wraps a loaded Coqui XTTS model with Auralis-style optimizations."""

    def __init__(self, tts):
        self.tts = tts
        self.model = tts.synthesizer.tts_model
        self.cache = ConditioningCache()
        self._lock = threading.Lock()  # XTTS on CPU is not re-entrant

    # -- conditioning (cached) ------------------------------------------------
    def _latents_for(self, speaker=None, speaker_wav=None):
        m = self.model
        if speaker is not None and speaker in m.speaker_manager.speakers:
            key = ConditioningCache._key_speaker(speaker)

            def _compute():
                gcl, se = m.speaker_manager.speakers[speaker].values()
                return gcl, se

        elif speaker_wav:
            paths = speaker_wav if isinstance(speaker_wav, (list, tuple)) else [speaker_wav]
            key = ConditioningCache._key_files(paths)

            def _compute():
                voice = m.clone_voice(speaker_wav)
                return voice["gpt_conditioning_latents"], voice["speaker_embedding"]
        else:
            raise ValueError("Need speaker or speaker_wav")

        gcl, se = self.cache.get_or_compute(key, _compute)
        return gcl, se

    # -- per-sentence primitives ---------------------------------------------
    def _split(self, text: str) -> list[str]:
        try:
            sents = self.tts.synthesizer.split_into_sentences(text)
        except Exception:
            sents = None
        return [s for s in (sents or []) if s.strip()] or [text]

    def _gpt_latents(self, sent: str, lang: str, gcl: torch.Tensor, se: torch.Tensor):
        """Run GPT generate + latent extraction for one sentence (the 87% cost)."""
        m = self.model
        sent = sent.strip().lower()
        lang = lang.split("-")[0]
        text_tokens = torch.IntTensor(m.tokenizer.encode(sent, lang=lang)).unsqueeze(0).to(m.device)
        gcl_d = gcl.to(m.device)
        with torch.no_grad():
            gpt_codes = m.gpt.generate(
                cond_latents=gcl_d,
                text_inputs=text_tokens,
                do_sample=True,
                top_p=0.85,
                top_k=50,
                temperature=0.75,
                num_return_sequences=m.gpt_batch_size,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=10.0,
                output_attentions=False,
            )
            expected_len = torch.tensor(
                [gpt_codes.shape[-1] * m.gpt.code_stride_len], device=text_tokens.device
            )
            text_len = torch.tensor([text_tokens.shape[-1]], device=m.device)
            gpt_latents = m.gpt(
                text_tokens, text_len, gpt_codes, expected_len,
                cond_latents=gcl_d, return_attentions=False, return_latent=True,
            )
        return gpt_latents, se.to(m.device)

    def _vocode(self, gpt_latents: torch.Tensor, se: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            wav = self.model.hifigan_decoder(gpt_latents, g=se).cpu().squeeze()
        return wav.numpy()

    # -- main entry -----------------------------------------------------------
    def synthesize(self, text: str, language: str, speaker=None, speaker_wav=None):
        """Return (np.ndarray waveform, stats).

        NOTE: Auralis pipelines the vocoder (HiFi-GAN) in a worker thread so it
        overlaps the next sentence's GPT decode. On an all-CPU Apple Silicon box
        that oversubscribes the cores and *slows down* long texts (measured
        0.66x), so we run the vocoder serially here and keep only the wins that
        measured positive: thread pinning + conditioning cache + upfront split.
        """
        with self._lock:
            gcl, se = self._latents_for(speaker, speaker_wav)
            sents = self._split(text)

            parts: list[np.ndarray] = []
            for sent in sents:
                latents, se_d = self._gpt_latents(sent, language, gcl, se)
                parts.append(self._vocode(latents, se_d))

            wav = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            stats = {"sentences": len(sents), "cached_conditioning": True}
            return wav, stats

    def synthesize_streaming(self, text, language, speaker=None, speaker_wav=None):
        """Yield each sentence's waveform as it completes (lower first-audio latency)."""
        with self._lock:
            gcl, se = self._latents_for(speaker, speaker_wav)
            sents = self._split(text)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = None
                for sent in sents:
                    latents, se_d = self._gpt_latents(sent, language, gcl, se)
                    if pending is not None:
                        yield pending.result()
                    pending = pool.submit(self._vocode, latents, se_d)
                if pending is not None:
                    yield pending.result()

    def save_wav(self, wav: np.ndarray, path: str, sample_rate: int = 24000):
        import soundfile as sf
        sf.write(path, wav, sample_rate)
