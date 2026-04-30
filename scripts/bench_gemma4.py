#!/usr/bin/env python3
"""Benchmark every installed gemma4* (and a control model) on Mira's prompts.

Usage::

    python scripts/bench_gemma4.py
    python scripts/bench_gemma4.py --models gemma4:e4b gemma4:26b
    python scripts/bench_gemma4.py --output .agent/runs/gemma4-bench.md

The script measures, per (model, prompt) pair:

- ``ttft_ms``   — time to the first response token (latency the user feels)
- ``total_ms``  — wall clock for the full response
- ``tokens``    — char count of the response (cheap stand-in for tokens)
- ``cps``       — chars per second (sustained throughput)

Five fixed prompts cover the surface Mira actually exercises:

1. Annotation (JSON, short)
2. Chat reply (free German prose)
3. Tool call (does the model emit a structured tool_call?)
4. JSON output stress (longer, strict schema)
5. Long context (loads a 1500-word passage as RAG context)

Output is a markdown table written next to the agent run logs. The script
prints the same table to stdout. It NEVER hits the network — it talks only
to the local Ollama at OLLAMA_BASE_URL (default ``http://localhost:11434``).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow ``python scripts/bench_gemma4.py`` from the repo root without install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.llm.ollama import OllamaProvider  # noqa: E402
from backend.llm.provider import ToolSchema  # noqa: E402
from backend.services.runtime_config import LlmOptions  # noqa: E402


# ─── Prompts ─────────────────────────────────────────────────────────────


_LONG_CONTEXT = (
    "Es war einmal in einem fernen Land ein König, der drei Töchter hatte. "
    "Die Jüngste war so schön, dass die Sonne selbst, die doch so vieles "
    "gesehen hat, sich verwunderte, sooft sie ihr ins Gesicht schien. "
) * 18  # ~1500 words


_PROMPTS: list[tuple[str, str]] = [
    (
        "annotation",
        'Erkläre das deutsche Wort „Ungeziefer" für einen B1-Lernenden. '
        'Antworte NUR als JSON: '
        '{"definition_de": "...", "definition_en": "...", "etymology": "..."}',
    ),
    (
        "chat",
        "Du bist ein einfühlsamer Literaturtutor. Beantworte kurz: "
        'Warum benutzt Kafka in „Die Verwandlung" das Wort "Ungeziefer" '
        "und nicht ein konkreteres Tier?",
    ),
    (
        "tool_call",
        "I have 12 minutes free. Pick a tool to start a short German drill "
        "session. Use the start_drill tool with competency_id='verbs.modal' "
        "and count=5.",
    ),
    (
        "json_strict",
        "Analyse den Satz: Als Gregor Samsa eines Morgens aus unruhigen "
        "Traeumen erwachte, fand er sich in seinem Bett zu einem "
        "ungeheueren Ungeziefer verwandelt. Antworte als JSON: "
        '{"subject":"...", "verb":"...", "tense":"...", "mood":"...", '
        '"gloss_en":"..."}.',
    ),
    (
        "long_context",
        f"Hier ist ein Märchenanfang:\n\n{_LONG_CONTEXT}\n\n"
        "Fasse den Text in genau zwei Sätzen auf Englisch zusammen.",
    ),
]


_TOOLS: list[ToolSchema] = [
    ToolSchema(
        name="start_drill",
        description="Start a vocabulary drill on the given competency.",
        parameters={
            "type": "object",
            "properties": {
                "competency_id": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["competency_id", "count"],
        },
    )
]


# ─── Result types + measurement ──────────────────────────────────────────


@dataclass
class Measurement:
    model: str
    prompt: str
    ttft_ms: Optional[float]
    total_ms: float
    chars: int
    tool_call: bool
    error: Optional[str] = None

    @property
    def cps(self) -> Optional[float]:
        if self.total_ms <= 0 or self.chars == 0:
            return None
        return round(self.chars / (self.total_ms / 1000.0), 1)


async def _run_one(
    provider: OllamaProvider,
    model: str,
    prompt_id: str,
    prompt: str,
    *,
    timeout: float = 120.0,
) -> Measurement:
    options = LlmOptions(model=model, think=False, temperature=0.3)
    if prompt_id == "tool_call":
        return await _run_tool_call(provider, model, prompt_id, prompt, timeout)

    start = time.perf_counter()
    ttft: Optional[float] = None
    chars = 0
    try:
        async for chunk in provider.stream_chat(
            [{"role": "user", "content": prompt}],
            options=options,
            timeout=timeout,
        ):
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            chars += len(chunk)
    except Exception as exc:  # pragma: no cover — best effort
        return Measurement(
            model=model,
            prompt=prompt_id,
            ttft_ms=None,
            total_ms=(time.perf_counter() - start) * 1000,
            chars=0,
            tool_call=False,
            error=str(exc),
        )

    return Measurement(
        model=model,
        prompt=prompt_id,
        ttft_ms=round(ttft, 1) if ttft is not None else None,
        total_ms=round((time.perf_counter() - start) * 1000, 1),
        chars=chars,
        tool_call=False,
    )


async def _run_tool_call(
    provider: OllamaProvider,
    model: str,
    prompt_id: str,
    prompt: str,
    timeout: float,
) -> Measurement:
    options = LlmOptions(model=model, think=False, temperature=0.0)
    start = time.perf_counter()
    ttft: Optional[float] = None
    chars = 0
    saw_tool_call = False
    try:
        async for ev in provider.chat_with_tools(
            [{"role": "user", "content": prompt}],
            tools=_TOOLS,
            options=options,
            timeout=timeout,
        ):
            if ttft is None and ev.kind.value in ("text", "tool_call"):
                ttft = (time.perf_counter() - start) * 1000
            if ev.kind.value == "text" and ev.text:
                chars += len(ev.text)
            elif ev.kind.value == "tool_call":
                saw_tool_call = True
                chars += len(json.dumps(ev.tool_args))
    except Exception as exc:  # pragma: no cover
        return Measurement(
            model=model,
            prompt=prompt_id,
            ttft_ms=None,
            total_ms=(time.perf_counter() - start) * 1000,
            chars=0,
            tool_call=False,
            error=str(exc),
        )

    return Measurement(
        model=model,
        prompt=prompt_id,
        ttft_ms=round(ttft, 1) if ttft is not None else None,
        total_ms=round((time.perf_counter() - start) * 1000, 1),
        chars=chars,
        tool_call=saw_tool_call,
    )


# ─── Markdown report ─────────────────────────────────────────────────────


def _render_markdown(measurements: list[Measurement], models: list[str]) -> str:
    """Render a per-model section followed by a wide cross-model summary."""
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append(f"# Gemma 4 benchmark — {now}")
    lines.append("")
    lines.append(
        "Generated by `scripts/bench_gemma4.py`. All numbers are for the "
        "local Ollama daemon — no network calls leave the machine."
    )
    lines.append("")
    lines.append(
        "Columns: ttft_ms = time-to-first-token, total_ms = full response "
        "time, chars = response length, cps = chars per second, tool = "
        "did the model emit a tool_call event?"
    )
    lines.append("")

    for model in models:
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| prompt | ttft_ms | total_ms | chars | cps | tool | error |")
        lines.append("|--------|---------:|----------:|-------:|------:|:------:|-------|")
        for m in [m for m in measurements if m.model == model]:
            tool_cell = "yes" if m.tool_call else ("n/a" if m.prompt != "tool_call" else "no")
            err = "" if not m.error else m.error[:60]
            lines.append(
                f"| {m.prompt} | {m.ttft_ms or '—'} | {m.total_ms} | "
                f"{m.chars} | {m.cps or '—'} | {tool_cell} | {err} |"
            )
        lines.append("")

    return "\n".join(lines)


# ─── Entry point ─────────────────────────────────────────────────────────


async def _async_main(args: argparse.Namespace) -> int:
    provider = OllamaProvider()

    if args.models:
        models = list(args.models)
    else:
        installed = await provider.list_models(timeout=5.0)
        models = [m for m in installed if m.lower().startswith("gemma4")]
        if not models:
            print(
                "No gemma4* models found via Ollama. Available:",
                ", ".join(installed) or "(none)",
                file=sys.stderr,
            )
            return 2

    print(f"Benchmarking models: {', '.join(models)}")
    measurements: list[Measurement] = []
    for model in models:
        for prompt_id, prompt in _PROMPTS:
            print(f"  [{model}] {prompt_id} ...", flush=True)
            m = await _run_one(provider, model, prompt_id, prompt, timeout=args.timeout)
            measurements.append(m)
            tail = (
                f"  -> total_ms={m.total_ms} chars={m.chars}"
                + (f" tool=yes" if m.tool_call else "")
                + (f" ERROR={m.error}" if m.error else "")
            )
            print(tail, flush=True)

    report = _render_markdown(measurements, models)

    out_path = Path(args.output) if args.output else Path(
        ROOT,
        ".agent",
        "runs",
        f"gemma4-bench-{dt.date.today().isoformat()}.md",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print()
    print(report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        help="Explicit model tags. Defaults to every gemma4* tag found.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-prompt timeout in seconds (default 120).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default .agent/runs/gemma4-bench-YYYY-MM-DD.md).",
    )
    args = parser.parse_args()

    # Honour OLLAMA_BASE_URL exactly like the runtime does.
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
