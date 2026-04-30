"""Ollama provider — concrete LLMProvider over a local Ollama HTTP server.

The agent and the legacy ``llm_service`` both go through this module. We talk
to two endpoints:

- ``/api/generate``  — one-shot completion (used by ``annotate_word``)
- ``/api/chat``      — streamed multi-turn chat (used by Q&A and the agent
                       loop). Tool calling is enabled by passing the
                       ``tools=`` field; Ollama 0.4+ forwards it to the
                       chat template for tool-capable model families
                       (Gemma 4, Llama 3.x, Qwen 2.5, etc.).

Failures are SWALLOWED here: ``generate`` returns ``None``; ``stream_chat``
yields nothing; ``chat_with_tools`` yields a single ``ProviderEvent.error_event``
followed by ``done()``. Callers are responsible for offline fallbacks.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, AsyncIterator, Optional

import httpx

from backend.config import get_settings
from backend.llm.provider import (
    LLMProvider,
    ProbeReport,
    ProviderEvent,
    ToolSchema,
)
from backend.services.runtime_config import LlmOptions

logger = logging.getLogger(__name__)


# ─── Think-block stripping ───────────────────────────────────────────────
#
# Some Ollama-served models (qwen3 thinking, deepseek-r1) emit hidden
# reasoning between <think> and </think>. We strip these client-side so the
# UI never shows them. The regex variant is used for non-streaming responses
# (annotate_word); the state-machine variant handles streaming where a tag
# may straddle two chunks. Both are also re-exported from
# backend.services.llm_service for the legacy tests.


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove every complete ``<think>…</think>`` block from a response."""
    return _THINK_BLOCK_RE.sub("", text).lstrip()


class ThinkStripper:
    """Incremental ``<think>…</think>`` filter for streaming tokens.

    Ollama's ``think=false`` flag suppresses thinking on models that support
    the parameter. For models that ignore it (or older Ollama versions) we
    still want clean UI output, so we maintain a tiny state machine:

    - When we see ``<think>`` we swallow tokens until ``</think>`` closes.
    - Short partial matches at a chunk boundary are buffered until we know
      whether they're the start of a thinking tag.
    """

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        out: list[str] = []
        buf = self._pending + chunk
        self._pending = ""

        while buf:
            if self._inside:
                end = buf.find("</think>")
                if end == -1:
                    keep = min(len(buf), 8)
                    self._pending = buf[-keep:]
                    buf = ""
                    break
                buf = buf[end + len("</think>") :]
                self._inside = False
                continue

            start = buf.find("<think>")
            if start == -1:
                tail_keep = 0
                for i in range(min(7, len(buf)), 0, -1):
                    if "<think>".startswith(buf[-i:]):
                        tail_keep = i
                        break
                if tail_keep:
                    out.append(buf[:-tail_keep])
                    self._pending = buf[-tail_keep:]
                else:
                    out.append(buf)
                buf = ""
                break

            out.append(buf[:start])
            buf = buf[start + len("<think>") :]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        leftover = self._pending
        self._pending = ""
        if self._inside:
            return ""
        return leftover


# ─── Missing-model detection (so we can hint `ollama pull <x>`) ──────────


_MODEL_NOT_FOUND_RE = re.compile(
    r"model ['\"]?([^'\"]+?)['\"]? not found|try pulling", re.IGNORECASE
)


def detect_missing_model(body: str) -> Optional[str]:
    """Return the missing model name if Ollama's error body reports one."""
    if not body:
        return None
    match = _MODEL_NOT_FOUND_RE.search(body)
    if not match:
        return None
    return match.group(1)


# ─── Audio-refusal detection (text-only models silently dropping audio) ──


# Regex catches the most common phrasings that text-only models produce when
# Ollama silently drops the ``audio`` field they cannot consume. We err on the
# side of false-positives because the worst case is one extra Whisper call;
# a false-negative is a refusal masquerading as a transcript (the dev-log bug
# from 2026-04-30). Patterns are case-insensitive and Unicode-aware.
_AUDIO_REFUSAL_RE = re.compile(
    r"(?:"
    r"kann\s+keine\s+audio"          # "Ich kann keine Audio-Dateien…"
    r"|kann\s+(?:keine|kein)\s+ton"  # "kann keinen Ton hören"
    r"|kann\s+nicht\s+h(?:ö|oe)ren"  # "kann nicht hören"
    r"|cannot\s+(?:hear|listen|process\s+audio)"
    r"|can(?:'|\u2019)?t\s+(?:hear|listen|process\s+audio)"
    r"|unable\s+to\s+(?:hear|listen|transcribe|process\s+audio)"
    r"|no\s+audio\s+(?:file|input|attached)"
    r"|please\s+(?:provide|give)\s+(?:me\s+)?(?:the\s+)?text"
    r"|geben\s+sie\s+mir\s+den\s+text"  # "geben Sie mir den Text"
    r"|gib\s+mir\s+den\s+text"
    r"|h(?:ö|oe)rt?\s+(?:keine|nichts)"
    r")",
    re.IGNORECASE,
)


def _looks_like_audio_refusal(text: str) -> bool:
    """Heuristic: does this look like a text-only model declining audio?"""
    if not text:
        return False
    return _AUDIO_REFUSAL_RE.search(text) is not None


# ─── OllamaProvider ──────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Concrete LLMProvider talking to a local Ollama daemon over HTTP."""

    name: str = "ollama"

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base_url_override = base_url

    @property
    def base_url(self) -> str:
        if self._base_url_override is not None:
            return self._base_url_override.rstrip("/")
        return get_settings().ollama_base_url.rstrip("/")

    # ── one-shot completion ────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        *,
        options: LlmOptions,
        timeout: float = 20.0,
    ) -> Optional[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": options.model,
            "prompt": prompt,
            "stream": False,
            "think": options.think,
            "options": {"temperature": options.temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    body = response.text
                    missing = detect_missing_model(body)
                    if missing:
                        logger.warning(
                            "Ollama model '%s' is not installed. "
                            "Run `ollama pull %s` or update OLLAMA_MODEL in .env.",
                            missing,
                            missing,
                        )
                    else:
                        logger.warning(
                            "Ollama returned %s: %s",
                            response.status_code,
                            body.strip()[:500],
                        )
                    return None
                data = response.json()
                return data.get("response", "").strip()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Ollama call failed: %s", exc)
            return None

    # ── streamed chat (text only) ──────────────────────────────────────

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        options: LlmOptions,
        timeout: float = 60.0,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": options.model,
            "stream": True,
            "think": options.think,
            "messages": messages,
            "options": {"temperature": options.temperature},
        }
        stripper = ThinkStripper()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        try:
                            raw = await response.aread()
                            body = raw.decode("utf-8", errors="replace")
                        except Exception:  # pragma: no cover
                            body = ""
                        logger.warning(
                            "Ollama chat returned %s: %s",
                            response.status_code,
                            body.strip()[:500],
                        )
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("message") or {}).get("content") or ""
                        if delta:
                            clean = stripper.feed(delta)
                            if clean:
                                yield clean
                        if obj.get("done"):
                            tail = stripper.flush()
                            if tail:
                                yield tail
                            return
                    tail = stripper.flush()
                    if tail:
                        yield tail
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Ollama chat stream failed: %s", exc)

    # ── streamed chat WITH tool calling ────────────────────────────────

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSchema],
        *,
        options: LlmOptions,
        timeout: float = 60.0,
    ) -> AsyncIterator[ProviderEvent]:
        """Streamed chat with native Ollama tool calls.

        Ollama emits tool calls inside the streamed message as
        ``message.tool_calls = [{function: {name, arguments}}, ...]``. We
        translate each into a single ``ProviderEvent.tool_call`` event so the
        controller can dispatch without knowing the wire format.
        """
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": options.model,
            "stream": True,
            "think": options.think,
            "messages": messages,
            "options": {"temperature": options.temperature},
            "tools": [t.to_openai() for t in tools],
        }
        stripper = ThinkStripper()

        # NOTE on the no-finally-yield pattern (PIVOT_ROADMAP §6.3, this fix):
        # An async generator that yields inside ``finally`` will raise
        # ``RuntimeError: async generator ignored GeneratorExit`` whenever the
        # consumer calls ``aclose()`` (which happens both on early break and
        # at GC time). We therefore emit the terminal ``done`` event from
        # each explicit completion path and leave ``finally`` for non-yielding
        # cleanup only — see backend/agent/controller.turn for the matching
        # discipline on the consumer side.
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        try:
                            raw = await response.aread()
                            body = raw.decode("utf-8", errors="replace")
                        except Exception:  # pragma: no cover
                            body = ""
                        missing = detect_missing_model(body)
                        msg = (
                            f"model not installed: {missing}"
                            if missing
                            else f"HTTP {response.status_code}: {body.strip()[:200]}"
                        )
                        yield ProviderEvent.error_event(msg)
                        yield ProviderEvent.done()
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        message = obj.get("message") or {}
                        delta = message.get("content") or ""
                        if delta:
                            clean = stripper.feed(delta)
                            if clean:
                                yield ProviderEvent.text_chunk(clean)
                        for tc in message.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            name = fn.get("name") or ""
                            raw_args = fn.get("arguments")
                            args: dict[str, Any]
                            if isinstance(raw_args, dict):
                                args = raw_args
                            elif isinstance(raw_args, str):
                                try:
                                    args = json.loads(raw_args)
                                except json.JSONDecodeError:
                                    args = {}
                            else:
                                args = {}
                            if name:
                                yield ProviderEvent.tool_call(
                                    name=name,
                                    args=args,
                                    call_id=tc.get("id"),
                                )
                        if obj.get("done"):
                            tail = stripper.flush()
                            if tail:
                                yield ProviderEvent.text_chunk(tail)
                            yield ProviderEvent.done()
                            return
                    tail = stripper.flush()
                    if tail:
                        yield ProviderEvent.text_chunk(tail)
                    yield ProviderEvent.done()
                    return
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Ollama chat_with_tools failed: %s", exc)
            yield ProviderEvent.error_event(str(exc))
            yield ProviderEvent.done()
            return

    # ── audio-multimodal STT (B.5, best-effort) ───────────────────────

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        content_type: str = "audio/wav",
        language: str = "de",
        timeout: float = 20.0,
    ) -> Optional[str]:
        """Try ``/api/generate`` with the audio attached as base64.

        Ollama's chat / generate endpoints accept ``audio: [<b64>]`` for
        models that advertise audio multimodality. Most Gemma 4 builds
        in the wild do not (yet) — Ollama silently drops the ``audio``
        field for text-only models, the model answers the bare prompt,
        and politely refuses ("Ich kann keine Audio-Dateien anhängen…").
        We detect that capability-denial pattern and return ``None`` so
        the caller falls through to faster-whisper instead of treating
        the refusal as a real transcript.
        """
        import base64

        if not audio_bytes:
            return None
        from backend.services.runtime_config import get_llm_options  # noqa: WPS433

        options = get_llm_options()
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        prompt = (
            "Transcribe the attached audio verbatim into "
            f"{language}. Output ONLY the transcript, no commentary."
        )
        payload: dict[str, Any] = {
            "model": options.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "audio": [b64],
            "options": {"temperature": 0.0},
        }
        url = f"{self.base_url}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    body = response.text
                    logger.info(
                        "Ollama audio generate returned %s; falling back to whisper.",
                        response.status_code,
                    )
                    if "audio" in body.lower() or "modality" in body.lower():
                        return None
                    return None
                data = response.json()
                text = (data.get("response") or "").strip()
                if not text or len(text) > 10_000:
                    return None
                if _looks_like_audio_refusal(text):
                    logger.info(
                        "Ollama model declined audio input "
                        "(no audio encoder?); falling back."
                    )
                    return None
                return text
        except (httpx.HTTPError, ValueError) as exc:
            logger.info("Ollama audio generate failed: %s", exc)
            return None

    # ── image-multimodal description (B.7, best-effort) ───────────────

    async def describe_image(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        timeout: float = 30.0,
    ) -> Optional[str]:
        """Run a strict-prompt vision call. Returns the raw model text."""
        import base64

        if not image_bytes:
            return None
        from backend.services.runtime_config import get_llm_options  # noqa: WPS433

        options = get_llm_options()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": options.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "images": [b64],
            "options": {"temperature": 0.1},
        }
        url = f"{self.base_url}/api/generate"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    body = response.text
                    logger.info(
                        "Ollama image generate returned %s: %s",
                        response.status_code,
                        body.strip()[:200],
                    )
                    return None
                data = response.json()
                text = (data.get("response") or "").strip()
                return text or None
        except (httpx.HTTPError, ValueError) as exc:
            logger.info("Ollama image generate failed: %s", exc)
            return None

    # ── model list + probe ─────────────────────────────────────────────

    async def list_models(self, *, timeout: float = 3.0) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("list_models failed: %s", exc)
            return []
        return [m.get("name") for m in data.get("models", []) if m.get("name")]

    async def probe(self, *, timeout: float = 3.0) -> ProbeReport:
        """Cheap availability report. Active model is read at call time."""
        from backend.services.runtime_config import get_llm_options

        active = get_llm_options().model
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            models = [
                m.get("name") for m in data.get("models", []) if m.get("name")
            ]
            return ProbeReport(
                reachable=True,
                models=models,
                configured=active,
                configured_available=active in models,
                error=None,
            )
        except Exception as exc:  # pragma: no cover — best effort
            return ProbeReport(
                reachable=False,
                models=[],
                configured=active,
                configured_available=False,
                error=str(exc),
            )


# ─── Process-wide singleton accessor ─────────────────────────────────────


_provider_lock = threading.Lock()
_provider: Optional[OllamaProvider] = None


def get_provider() -> OllamaProvider:
    """Return the process-wide provider instance.

    A single provider is reused so that future implementations can amortise
    expensive setup (model download checks, gRPC connections for LiteRT,
    etc.). The current Ollama provider is stateless — the lock is just
    forward-compatible.
    """
    global _provider
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is None:
            _provider = OllamaProvider()
        return _provider


def reset_for_tests() -> None:
    """Drop the cached provider — pytest use only."""
    global _provider
    with _provider_lock:
        _provider = None
