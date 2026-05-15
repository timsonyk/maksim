"""Claude brain — streaming + tool-use loop.

Exposes `stream_turn(user_text)` as an async generator that yields text
fragments (already split into spoken sentences) destined for TTS. Tool
calls are executed transparently — the LLM keeps running until it stops
calling tools, and only natural-language tokens are emitted upstream."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import anthropic

from . import log, tools
from .config import settings


_SENTENCE_END = (".", "!", "?", "\n", "…", ":")


def _split_sentences(buf: str) -> tuple[list[str], str]:
    """Split text into spoken segments. A segment ends at sentence
    punctuation followed by space/newline, OR exceeds ~60 chars at a comma.
    Returns (complete_sentences, leftover)."""
    out: list[str] = []
    last_cut = 0
    i = 0
    while i < len(buf):
        c = buf[i]
        nxt = buf[i + 1] if i + 1 < len(buf) else ""
        if c in _SENTENCE_END and (nxt in (" ", "\n", "") or i == len(buf) - 1):
            seg = buf[last_cut : i + 1].strip()
            if seg:
                out.append(seg)
            last_cut = i + 1
        elif c == "," and i - last_cut > 60 and nxt == " ":
            seg = buf[last_cut : i + 1].strip()
            if seg:
                out.append(seg)
            last_cut = i + 1
        i += 1
    return out, buf[last_cut:]


class Brain:
    MAX_HISTORY = 12  # turns

    def __init__(self, api_key: str, model: str, max_tokens: int,
                 temperature: float, persona: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.persona = persona
        self.history: List[dict] = []

    def _trim_history(self) -> None:
        """Trim to MAX_HISTORY entries, but never split a tool_use/tool_result
        pair (Claude rejects messages that start with an orphan tool_result, or
        contain a tool_use not immediately followed by its tool_result)."""
        if len(self.history) <= self.MAX_HISTORY:
            return
        target_start = len(self.history) - self.MAX_HISTORY
        # Advance to the first text-only user message — that's a safe cut point.
        while target_start < len(self.history):
            msg = self.history[target_start]
            if msg["role"] == "user" and isinstance(msg["content"], str):
                break
            target_start += 1
        self.history = self.history[target_start:]

    def reset(self) -> None:
        """Drop conversation history (e.g. after a 400 from a desync)."""
        self.history = []

    async def stream_turn(self, user_text: str) -> AsyncIterator[str]:
        """Yield spoken sentences for one user turn."""
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        log.event("llm.user", text=user_text)

        # Inner generator so we can intercept tool_use/tool_result desync errors,
        # reset history, and retry from scratch as if this were turn #1.
        try:
            async for s in self._stream_inner(user_text):
                yield s
        except anthropic.BadRequestError as e:
            msg = str(e)
            if "tool_use" in msg and "tool_result" in msg:
                log.event("llm.history_desync_recovery")
                self.reset()
                self.history.append({"role": "user", "content": user_text})
                async for s in self._stream_inner(user_text):
                    yield s
            else:
                raise

    async def _stream_inner(self, user_text: str) -> AsyncIterator[str]:

        text_buf = ""
        # Tool-use loop: keep streaming until the assistant stops calling tools.
        while True:
            assistant_blocks: list[dict] = []
            tool_uses: list[dict] = []
            stop_reason = None

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.persona,
                tools=tools.TOOLS_SCHEMA,
                messages=self.history,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            text_buf += event.delta.text
                            sentences, text_buf = _split_sentences(text_buf)
                            for s in sentences:
                                yield s
                final = await stream.get_final_message()
                stop_reason = final.stop_reason
                for block in final.content:
                    if block.type == "text":
                        assistant_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                        tool_uses.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

            # commit assistant message to history
            self.history.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use" or not tool_uses:
                # flush remainder
                if text_buf.strip():
                    yield text_buf.strip()
                    text_buf = ""
                break

            # execute all requested tools in parallel
            results = await asyncio.gather(*[
                tools.dispatch(tu["name"], tu["input"]) for tu in tool_uses
            ])
            tool_result_blocks = []
            for tu, res in zip(tool_uses, results):
                # Vision tools (look_at_screen) return {"_image": "<b64>", "_media_type": ...}.
                # Convert that into a tool_result whose content is a list with an
                # image block so the model actually sees the screenshot.
                if isinstance(res, dict) and "_image" in res:
                    content = [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": res.get("_media_type", "image/jpeg"),
                            "data": res["_image"],
                        }},
                        {"type": "text", "text": f"screenshot captured ({res.get('size_bytes', 0)} bytes)"},
                    ]
                    log.event("llm.tool_call", name=tu["name"], input=tu["input"],
                              result_preview="<image>")
                else:
                    content = str(res)[:4000]
                    log.event("llm.tool_call", name=tu["name"], input=tu["input"],
                              result_preview=str(res)[:300])
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": content,
                })
            self.history.append({"role": "user", "content": tool_result_blocks})
            self._trim_history()
            # continue the loop — model will now respond to tool results


def make_brain() -> Brain:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    return Brain(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        persona=settings.persona,
    )
