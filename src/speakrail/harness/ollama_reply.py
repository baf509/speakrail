"""Translate a live Speakrail context into chat messages and stream an Ollama reply."""

import json

from . import protocol as P


def messages(ctx: P.Context, system: str) -> list[dict]:
    result = [{"role": "system", "content": system + "\n"
               "Speakrail handles turn-taking. When called, respond now in plain spoken language. "
               "Bracketed tool results in the transcript are background context."}]
    role = "user"
    parts = []

    def finish():
        if parts:
            result.append({"role": role, "content": " ".join(parts)})
            parts.clear()

    for start, end, kind in ctx.spans:
        if kind not in {"harness", "word", "reply", "reply_noloss", "interject", "interject_noloss", "tool_result"}:
            continue
        text = P.decode(ctx.ids[start:end]).strip()
        if kind == "harness":
            if text.endswith(P.USER_OPEN.strip()):
                finish()
                role = "user"
            elif text.endswith(P.MODEL_OPEN.strip()):
                finish()
                role = "assistant"
        elif kind.startswith("interject"):
            finish()
            result.append({"role": "assistant", "content": text})
        elif kind == "tool_result":
            text = text.replace("<|tool_response>", "").replace("<tool_response|>", "\n").replace(P.Q, '"').strip()
            parts.append(f"[Tool result: {text}]")
        elif text:
            parts.append(text)
    finish()
    if result[-1]["role"] != "user":
        result.append({"role": "user", "content": "Continue the conversation."})
    return result


async def stream(http, url, model, messages, tools, max_tokens):
    body = {"model": model, "messages": messages, "tools": tools, "stream": True, "think": False,
            "options": {"num_predict": max_tokens, "num_ctx": 4096}}
    async with http.post(url.rstrip("/") + "/api/chat", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"Ollama HTTP {response.status}: {(await response.text())[:300]}")
        async for line in response.content:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"Ollama: {chunk['error']}")
            yield chunk
            if chunk.get("done"):
                return
        raise RuntimeError("Ollama disconnected before finishing its reply")


async def openai_stream(http, url, model, messages, tools, max_tokens, headers, thinking_budget=0):
    body = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens,
            "reasoning_effort": "none"}
    if thinking_budget > 0:
        body["thinking_budget"] = thinking_budget      # NInfer: caps the thinking block it would open anyway
    if tools:
        body["tools"] = tools
    calls = {}
    finished = False
    async with http.post(url, json=body, headers=headers, allow_redirects=False) as response:
        if response.status != 200:
            raise RuntimeError(f"Reply model HTTP {response.status}: {(await response.text())[:300]}")
        async for line in response.content:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                finished = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(f"Reply model: {chunk['error']}")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if text := delta.get("content"):
                    yield {"message": {"content": text}}
                for tool in delta.get("tool_calls") or []:
                    call = calls.setdefault(tool["index"], {"id": "", "type": "function",
                                                           "function": {"name": "", "arguments": ""}})
                    if tool.get("id"):
                        call["id"] = tool["id"]
                    for field in ("name", "arguments"):
                        call["function"][field] += (tool.get("function") or {}).get(field) or ""
                finished |= bool(choice.get("finish_reason"))
            if usage := chunk.get("usage"):
                yield {"eval_count": usage.get("completion_tokens")}
        if not finished:
            raise RuntimeError("Reply model disconnected before finishing its reply")
        if calls:
            yield {"message": {"tool_calls": [calls[index] for index in sorted(calls)]}}
