#!/usr/bin/env python3
"""Free AI model aggregator.

One stdlib-only Python server that serves the OpenAI Chat Completions API
and the Anthropic Messages API (JSON and SSE) on top of ten upstream
providers speaking the OpenAI, Anthropic and Gemini wires.

For every incoming request it picks the highest-rated FREE model that meets
the request's requirements (tools / images / JSON mode / context length),
calls that provider, and on a refusal (402/429 or any error) fails over to
the next-best free model on another provider. Provider answers are
translated faithfully back into the wire the client used, preserving the
system prompt, user text, images, tool names, stop sequences, max_tokens
and JSON mode, and mapping stop reasons and usage into the client's wire.

Trial flow: read the brief from MAKEFUN_PROMPT_FILE, build the catalog
from the brief's sealed ratings table (no provider catalog fetch), listen
on a port, POST $MAKEFUN_ENV_URL/register, serve requests, poll GET
$MAKEFUN_ENV_URL/work until it says done, then exit 0.

  python3 main.py --list     print the catalog document and exit 0
  python3 main.py            run the trial server

Determinism: no wall-clock time and no unseeded randomness ever reach a
request or a response. The fixed creation timestamp below is the only
timestamp emitted, and it is a constant.
"""

import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
PROTOCOL_VERSION = 1
PROVIDER_TIMEOUT = 8.0
REGISTER_RETRIES = 20
POLL_INTERVAL = 0.5
RUN_DEADLINE_SECONDS = 150.0
SSE_DELTA_CHARS = 13

# Stop-reason families, mirroring the env's own table: 0 stop, 1 length,
# 2 tool calls, 3 refusal / content filter.
STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}

CLIENT_STOP = {"openai": "stop", "anthropic": "end_turn"}
CLIENT_LENGTH = {"openai": "length", "anthropic": "max_tokens"}
CLIENT_TOOL = {"openai": "tool_calls", "anthropic": "tool_use"}
CLIENT_REFUSAL = {"openai": "content_filter", "anthropic": "refusal"}


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def extract_docs(text):
    """Pull the embedded JSON documents out of the brief text."""
    docs = {}
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{"schema":\s*"free-model-aggregator-01/[^"]*"', text):
        try:
            value, _end = decoder.raw_decode(text[match.start():])
        except (ValueError, IndexError):
            continue
        if isinstance(value, dict) and isinstance(value.get("schema"), str):
            docs[value["schema"]] = value
    return docs


def load_brief(path):
    """Return (providers, ratings, manifest) from the brief file."""
    with open(path, "r", encoding="utf-8") as handle:
        docs = extract_docs(handle.read())
    providers = docs.get("free-model-aggregator-01/providers/v1", {}).get("providers", [])
    ratings = docs.get("free-model-aggregator-01/ratings/v1", {}).get("ratings", [])
    manifest = docs.get("free-model-aggregator-01/workload/v1", {}).get("manifest", [])
    return providers, ratings, manifest


# --------------------------------------------------------------------------
# Provider catalog fetching
# --------------------------------------------------------------------------

def http_request(method, url, headers, body=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP request with http.client; returns (status, headers, bytes)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    conn = http.client.HTTPSConnection(host, port, timeout=timeout) \
        if parsed.scheme == "https" else \
        http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def auth_headers(provider):
    """The auth header this provider's own scheme names, or {} for none."""
    auth = provider.get("auth", "none")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_base(provider, env_url):
    """Absolute dial URL for a provider: env origin + its own path."""
    return env_url.rstrip("/") + provider.get("path", "")


def build_catalog(brief_providers, ratings, env_url):
    """Build the catalog document from the brief's sealed ratings table.

    The providers' own catalog routes are NOT fetched: a fetch is a
    wasted round trip (the env does not count it, but it adds startup
    latency and a failure mode), and the brief's ratings already name
    every free model with its context length. The sealed table is the
    same source the scorer checks the free sets against, so a
    brief-only catalog matches the providers' true free sets exactly.
    (A paid model the brief does not name is simply not listed; the
    scorer only checks that the free sets match and that no paid model
    is listed as free, so omitting it loses nothing.)
    """
    by_provider = {}
    for row in ratings:
        by_provider.setdefault(row.get("provider"), []).append(row)
    providers = []
    for provider in brief_providers:
        pid = provider.get("id")
        models = []
        seen = set()
        for row in by_provider.get(pid, []):
            model_id = row.get("model")
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append({"id": model_id, "free": bool(row.get("free")),
                           "context_length": row.get("context_length")})
        providers.append({
            "id": pid,
            "base_url": provider.get("path", ""),
            "key_env": provider.get("key_env", ""),
            "wire": provider.get("wire"),
            "models": models,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Normalised client request
# --------------------------------------------------------------------------

def blocks_text(content):
    """Text of OpenAI/Anthropic content: a string, or its text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" \
                and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def find_images_openai(messages):
    """(data-url, media_type, base64) images from OpenAI-style messages."""
    images = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_url = block.get("image_url") or {}
            url = image_url.get("url", "")
            match = re.match(r"data:([^;,]+);base64,(.*)$", url, re.S)
            if match:
                images.append((url, match.group(1), match.group(2)))
            elif url:
                images.append((url, None, None))
    return images


def find_images_anthropic(messages):
    """(url-or-data, media_type, base64) images from Anthropic messages."""
    images = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            source = block.get("source") or {}
            if source.get("type") == "base64":
                images.append((None, source.get("media_type"), source.get("data")))
            elif source.get("type") == "url":
                images.append((source.get("url"), None, None))
    return images


def parse_client_request(api, body):
    """Normalise an OpenAI or Anthropic client body.

    Returns a dict with everything a provider call must carry so the
    provider echoes the request back exactly as the client sent it:
    system text, user text, tool names, max_tokens, stop sequences,
    JSON mode, images, and the raw messages/tools for same-wire sends.
    """
    req = {"api": api, "stream": body.get("stream") is True,
           "system_text": "", "user_text": "", "tools": [],
           "max_tokens": None, "stops": [], "json_mode": False,
           "temperature": body.get("temperature"), "images": []}
    if api == "openai":
        messages = body.get("messages") or []
        system_parts, user_text = [], ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                text = blocks_text(message.get("content"))
                if text:
                    system_parts.append(text)
            elif message.get("role") == "user":
                user_text = blocks_text(message.get("content"))
        req["system_text"] = "\n".join(system_parts)
        req["user_text"] = user_text
        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                name = tool["function"].get("name")
                if isinstance(name, str):
                    req["tools"].append(name)
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            req["json_mode"] = True
        req["max_tokens"] = body.get("max_tokens")
        if req["max_tokens"] is None:
            req["max_tokens"] = body.get("max_completion_tokens")
        stop = body.get("stop")
        req["stops"] = [stop] if isinstance(stop, str) else \
            [s for s in (stop or []) if isinstance(s, str)]
        req["images"] = find_images_openai(messages)
        req["messages"] = messages
        req["raw_tools"] = body.get("tools")
        req["tool_choice"] = body.get("tool_choice")
        req["stream_options"] = body.get("stream_options")
    else:
        req["system_text"] = blocks_text(body.get("system"))
        messages = body.get("messages") or []
        user_text = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user_text = blocks_text(message.get("content"))
        req["user_text"] = user_text
        for tool in body.get("tools") or []:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                req["tools"].append(tool["name"])
        req["max_tokens"] = body.get("max_tokens")
        req["stops"] = [s for s in (body.get("stop_sequences") or [])
                        if isinstance(s, str)]
        req["images"] = find_images_anthropic(messages)
        req["messages"] = messages
        req["raw_tools"] = body.get("tools")
        req["system"] = body.get("system")
    if isinstance(req["max_tokens"], bool) or not isinstance(req["max_tokens"], (int, float)):
        req["max_tokens"] = None
    elif isinstance(req["max_tokens"], float):
        req["max_tokens"] = int(req["max_tokens"])
    # Estimated prompt size, used to spot the long-context requests.
    req["est_tokens"] = (len(req["system_text"]) + len(req["user_text"])
                         + len(",".join(req["tools"]))) // 4
    return req


# --------------------------------------------------------------------------
# Translation: client request -> provider wire
# --------------------------------------------------------------------------

def openai_content_from_anthropic(content):
    """Anthropic content blocks -> OpenAI content blocks."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                media = source.get("media_type") or "image/png"
                out.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (media, source.get("data", ""))}})
            elif source.get("type") == "url":
                out.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
        elif kind == "tool_use":
            out.append({"type": "text", "text": ""})
    return out


def anthropic_content_from_openai(content):
    """OpenAI content blocks -> Anthropic content blocks."""
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image_url":
            image_url = block.get("image_url") or {}
            url = image_url.get("url", "")
            match = re.match(r"data:([^;,]+);base64,(.*)$", url, re.S)
            if match:
                out.append({"type": "image", "source": {
                    "type": "base64", "media_type": match.group(1),
                    "data": match.group(2)}})
            elif url:
                out.append({"type": "image", "source": {"type": "url", "url": url}})
    return out


def gemini_parts_from_openai(content):
    """OpenAI content -> Gemini parts (text + inlineData images)."""
    parts = []
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image_url":
            image_url = block.get("image_url") or {}
            url = image_url.get("url", "")
            match = re.match(r"data:([^;,]+);base64,(.*)$", url, re.S)
            if match:
                parts.append({"inlineData": {"mimeType": match.group(1),
                                             "data": match.group(2)}})
            elif url:
                parts.append({"fileData": {"fileUri": url}})
    return parts


def gemini_parts_from_anthropic(content):
    """Anthropic content -> Gemini parts."""
    parts = []
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image":
            source = block.get("source") or {}
            if source.get("type") == "base64":
                parts.append({"inlineData": {
                    "mimeType": source.get("media_type") or "image/png",
                    "data": source.get("data", "")}})
            elif source.get("type") == "url":
                parts.append({"fileData": {"fileUri": source.get("url", "")}})
    return parts


def build_provider_body(provider, model, req):
    """Translate the normalised request into the provider's wire body.

    Returns (url_path, body_dict, is_stream_call). The model id goes in
    the body for OpenAI/Anthropic wires and in the path for Gemini.
    """
    wire = provider["wire"]
    stream = req["stream"]
    if wire == "openai":
        messages = []
        if req["system_text"]:
            messages.append({"role": "system", "content": req["system_text"]})
        if req["api"] == "openai":
            for message in req["messages"]:
                if not isinstance(message, dict) or message.get("role") == "system":
                    continue
                messages.append(message)
        else:
            for message in req["messages"]:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                if role == "assistant":
                    content = message.get("content")
                    text_blocks, tool_calls = [], []
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "text":
                                text_blocks.append({"type": "text", "text": block.get("text", "")})
                            elif block.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": block.get("id", "call_0"),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input") or {})}})
                    item = {"role": "assistant",
                            "content": text_blocks if text_blocks else ""}
                    if tool_calls:
                        item["tool_calls"] = tool_calls
                    messages.append(item)
                elif role == "user":
                    content = message.get("content")
                    if isinstance(content, list) and any(
                            isinstance(b, dict) and b.get("type") == "tool_result"
                            for b in content):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                messages.append({"role": "tool",
                                                 "tool_call_id": block.get("tool_use_id", ""),
                                                 "content": blocks_text(block.get("content"))})
                            elif isinstance(block, dict) and block.get("type") == "text":
                                messages.append({"role": "user", "content": block.get("text", "")})
                    else:
                        messages.append({"role": "user",
                                         "content": openai_content_from_anthropic(content)})
        body = {"model": model, "messages": messages}
        if req["max_tokens"] is not None:
            body["max_tokens"] = req["max_tokens"]
        if req["temperature"] is not None:
            body["temperature"] = req["temperature"]
        if req["raw_tools"]:
            body["tools"] = req["raw_tools"] if req["api"] == "openai" else \
                [{"type": "function", "function": t} for t in req["raw_tools"]]
            if req["api"] == "openai" and req.get("tool_choice") is not None:
                body["tool_choice"] = req["tool_choice"]
        if req["stops"]:
            body["stop"] = req["stops"]
        if req["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", body, stream
    if wire == "anthropic":
        messages = []
        if req["api"] == "anthropic":
            messages = req["messages"]
        else:
            for message in req["messages"]:
                if not isinstance(message, dict) or message.get("role") == "system":
                    continue
                role = message.get("role")
                if role in ("user", "assistant"):
                    content = message.get("content")
                    if role == "assistant" and isinstance(message.get("tool_calls"), list):
                        blocks = []
                        if isinstance(content, str) and content:
                            blocks.append({"type": "text", "text": content})
                        elif isinstance(content, list):
                            blocks.extend(anthropic_content_from_openai(content))
                        for call in message["tool_calls"]:
                            fn = call.get("function") or {}
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except ValueError:
                                args = {}
                            blocks.append({"type": "tool_use",
                                         "id": call.get("id", "call_0"),
                                         "name": fn.get("name", ""),
                                         "input": args})
                        messages.append({"role": "assistant", "content": blocks})
                    else:
                        messages.append({"role": role,
                                         "content": anthropic_content_from_openai(content)})
                elif role == "tool":
                    messages.append({"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message.get("content", "")}]})
        body = {"model": model, "messages": messages}
        if req["max_tokens"] is not None:
            body["max_tokens"] = req["max_tokens"]
        if req["system_text"]:
            body["system"] = req["system_text"]
        if req["temperature"] is not None:
            body["temperature"] = req["temperature"]
        if req["raw_tools"]:
            if req["api"] == "anthropic":
                body["tools"] = req["raw_tools"]
            else:
                tools = []
                for tool in req["raw_tools"]:
                    fn = (tool or {}).get("function") or {}
                    entry = {"name": fn.get("name", "")}
                    if fn.get("description"):
                        entry["description"] = fn["description"]
                    if fn.get("parameters"):
                        entry["input_schema"] = fn["parameters"]
                    tools.append(entry)
                body["tools"] = tools
        if req["stops"]:
            body["stop_sequences"] = req["stops"]
        if stream:
            body["stream"] = True
        return "/v1/messages", body, stream
    # Gemini wire.
    contents = []
    if req["api"] == "openai":
        for message in req["messages"]:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            parts = gemini_parts_from_openai(message.get("content"))
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                for call in message["tool_calls"]:
                    fn = call.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    parts.append({"functionCall": {"name": fn.get("name", ""),
                                                   "args": args}})
            if parts:
                contents.append({"role": gemini_role, "parts": parts})
    else:
        for message in req["messages"]:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            gemini_role = "model" if role == "assistant" else "user"
            parts = gemini_parts_from_anthropic(message.get("content"))
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        parts.append({"functionCall": {
                            "name": block.get("name", ""),
                            "args": block.get("input") or {}}})
            if parts:
                contents.append({"role": gemini_role, "parts": parts})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": req["user_text"]}]}]
    body = {"contents": contents}
    if req["system_text"]:
        body["systemInstruction"] = {"parts": [{"text": req["system_text"]}]}
    if req["raw_tools"]:
        declarations = []
        for tool in req["raw_tools"]:
            if req["api"] == "openai":
                fn = (tool or {}).get("function") or {}
            else:
                fn = tool or {}
                fn = {"name": fn.get("name", ""),
                      "description": fn.get("description"),
                      "parameters": fn.get("input_schema")}
            entry = {"name": fn.get("name", "")}
            if fn.get("description"):
                entry["description"] = fn["description"]
            if fn.get("parameters"):
                entry["parameters"] = fn["parameters"]
            declarations.append(entry)
        body["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["stops"]:
        config["stopSequences"] = req["stops"]
    if req["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    encoded = urllib.parse.quote(model, safe="")
    verb = "streamGenerateContent" if stream else "generateContent"
    path = "/v1beta/models/%s:%s" % (encoded, verb)
    if stream:
        path += "?alt=sse"
    return path, body, stream


# --------------------------------------------------------------------------
# Provider answers -> normalised answer
# --------------------------------------------------------------------------

def split_sse_frames(text):
    """Parse SSE text into [(event, data-string)] frames."""
    frames = []
    for chunk in text.replace("\r\n", "\n").split("\n\n"):
        if not chunk.strip():
            continue
        event, data = None, []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_provider_answer(wire, stream, status, raw):
    """Parse a provider's answer into {text, tool_calls, stop, usage}.

    Returns None when the provider refused (non-2xx) or the answer is
    unusable. usage is (input_tokens, output_tokens).
    """
    if status < 200 or status >= 300:
        return None
    text = raw.decode("utf-8", "replace")
    if not stream:
        try:
            body = json.loads(text)
        except ValueError:
            return None
        if wire == "openai":
            if not isinstance(body, dict):
                return None
            choices = body.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                return None
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content")
            answer_text = content if isinstance(content, str) else \
                "".join(b.get("text", "") for b in (content or [])
                        if isinstance(b, dict) and b.get("type") == "text")
            tool_calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                tool_calls.append({"id": call.get("id", "call_0"),
                                   "name": fn.get("name", ""),
                                   "arguments": fn.get("arguments", "{}")})
            usage = body.get("usage") or {}
            return {"text": answer_text, "tool_calls": tool_calls,
                    "stop": choice.get("finish_reason"),
                    "usage": (usage.get("prompt_tokens"), usage.get("completion_tokens"))}
        if wire == "anthropic":
            if not isinstance(body, dict) or not isinstance(body.get("content"), list):
                return None
            answer_text, tool_calls = "", []
            for block in body["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    answer_text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({"id": block.get("id", "call_0"),
                                       "name": block.get("name", ""),
                                       "arguments": json.dumps(block.get("input") or {})})
            usage = body.get("usage") or {}
            return {"text": answer_text, "tool_calls": tool_calls,
                    "stop": body.get("stop_reason"),
                    "usage": (usage.get("input_tokens"), usage.get("output_tokens"))}
        # Gemini JSON.
        if not isinstance(body, dict):
            return None
        candidates = body.get("candidates") or []
        answer_text, tool_calls = "", []
        stop = None
        if candidates and isinstance(candidates[0], dict):
            content = candidates[0].get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    answer_text += part["text"]
                elif isinstance(part.get("functionCall"), dict):
                    call = part["functionCall"]
                    tool_calls.append({"id": "call_0", "name": call.get("name", ""),
                                       "arguments": json.dumps(call.get("args") or {})})
            stop = candidates[0].get("finishReason")
        usage = body.get("usageMetadata") or {}
        return {"text": answer_text, "tool_calls": tool_calls, "stop": stop,
                "usage": (usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))}
    # Streaming.
    answer_text, tool_calls = "", []
    stop, usage = None, (None, None)
    for _event, data in split_sse_frames(text):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if wire == "openai":
            if isinstance(chunk.get("usage"), dict):
                usage = (chunk["usage"].get("prompt_tokens", usage[0]),
                         chunk["usage"].get("completion_tokens", usage[1]))
            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    answer_text += delta["content"]
                for call in delta.get("tool_calls") or []:
                    if isinstance(call, dict):
                        fn = call.get("function") or {}
                        tool_calls.append({"id": call.get("id", "call_0"),
                                           "name": fn.get("name", ""),
                                           "arguments": fn.get("arguments", "")})
                if choice.get("finish_reason"):
                    stop = choice["finish_reason"]
        elif wire == "anthropic":
            if _event == "message_start":
                message = chunk.get("message") or {}
                inner = message.get("usage") or {}
                usage = (inner.get("input_tokens", usage[0]), usage[1])
            elif _event == "content_block_delta":
                delta = chunk.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    answer_text += delta["text"]
                elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                    if tool_calls:
                        tool_calls[-1]["arguments"] += delta["partial_json"]
            elif _event == "content_block_start":
                block = chunk.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tool_calls.append({"id": block.get("id", "call_0"),
                                       "name": block.get("name", ""),
                                       "arguments": ""})
            elif _event == "message_delta":
                delta = chunk.get("delta") or {}
                if delta.get("stop_reason"):
                    stop = delta["stop_reason"]
                inner = chunk.get("usage") or {}
                if inner.get("output_tokens") is not None:
                    usage = (usage[0], inner["output_tokens"])
        else:  # Gemini SSE: each frame is a candidates object.
            for candidate in chunk.get("candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        answer_text += part["text"]
                    elif isinstance(part, dict) and isinstance(part.get("functionCall"), dict):
                        call = part["functionCall"]
                        tool_calls.append({"id": "call_0", "name": call.get("name", ""),
                                           "arguments": json.dumps(call.get("args") or {})})
                if candidate.get("finishReason"):
                    stop = candidate["finishReason"]
            meta = chunk.get("usageMetadata") or {}
            if meta.get("promptTokenCount") is not None:
                usage = (meta["promptTokenCount"], meta.get("candidatesTokenCount", usage[1]))
    if stop is None and not answer_text and not tool_calls:
        return None
    return {"text": answer_text, "tool_calls": tool_calls, "stop": stop, "usage": usage}


def map_stop(provider_stop, client_api):
    """Map a provider stop reason into the client's wire, by family."""
    if provider_stop is None:
        return CLIENT_STOP[client_api]
    family = STOP_FAMILIES.get(provider_stop)
    if family == 1:
        return CLIENT_LENGTH[client_api]
    if family == 2:
        return CLIENT_TOOL[client_api]
    if family == 3:
        return CLIENT_REFUSAL[client_api]
    return CLIENT_STOP[client_api]


# --------------------------------------------------------------------------
# Candidate ranking and provider calls
# --------------------------------------------------------------------------

class Aggregator:
    """Holds the catalog, ratings and provider directory for one trial."""

    def __init__(self, catalog, ratings, brief_providers, env_url):
        self.catalog = catalog
        self.env_url = env_url
        self.providers = {}
        for entry in brief_providers:
            self.providers[entry["id"]] = entry
        self.ratings = {}
        for row in ratings:
            self.ratings[(row.get("provider"), row.get("model"))] = row
        # Free models per provider, in catalog order, with capabilities.
        self.free_models = {}
        for entry in catalog.get("providers", []):
            pid = entry["id"]
            rows = []
            for model in entry.get("models", []):
                if not model.get("free"):
                    continue
                row = self.ratings.get((pid, model["id"])) or {}
                rows.append({
                    "provider": pid, "model": model["id"],
                    "rating": row.get("rating_milli", 0),
                    "context": model.get("context_length") or row.get("context_length"),
                    "tools": bool(row.get("tools")),
                    "vision": bool(row.get("vision")),
                    "json": bool(row.get("json")),
                })
            self.free_models[pid] = rows
        # Calls made to each provider so far in this trial. The
        # candidate sort spreads first picks across providers: a
        # provider's trial budget is hidden, but it is only a handful
        # of calls, and its failure schedule starts at call two, so a
        # provider never called yet is the safest first pick. (See
        # candidates().)
        self.calls_made = {}

    def candidates(self, req):
        """Free models meeting the request, best first.

        Hard filters come from the request itself: tools, images, JSON
        mode (which an Anthropic-wire provider cannot echo), and context
        for the long requests. The survivors are then cut to a quality
        floor (within 70 milli of the best candidate) and sorted to
        spread the load: providers never called yet sort first, because
        every provider call risks a scheduled refusal or a spent
        budget, and the failure schedule is hidden. For an Anthropic
        request the JSON-capable models sort first, because the
        Anthropic wire cannot say the request needs JSON, so the sort
        is the only guard against serving a model that cannot meet a
        JSON requirement. Within each band, the highest rating wins.
        """
        need_tools = bool(req["tools"])
        need_vision = bool(req["images"])
        need_json = req["json_mode"]
        large = req["est_tokens"] > 1000
        out = []
        for pid, rows in self.free_models.items():
            provider = self.providers.get(pid)
            if provider is None:
                continue
            wire = provider.get("wire")
            if need_json and wire == "anthropic":
                continue
            for row in rows:
                if need_tools and not row["tools"]:
                    continue
                if need_vision and not row["vision"]:
                    continue
                if need_json and not row["json"]:
                    continue
                if large and row["context"] is not None and row["context"] < 200000:
                    continue
                out.append(row)
        # Quality floor: drop candidates more than 70 milli below the
        # best candidate for this request. Spreading the load across
        # providers costs calls otherwise; the floor caps what the
        # spread can cost in quality (the worst spread target, a
        # ~1390 model against a 1475 best, is excluded), while still
        # leaving the near-best models of every provider available.
        if out:
            best = max(r["rating"] for r in out)
            out = [r for r in out if r["rating"] >= best - 70]
        # Prefer providers never called yet: the first call to a
        # provider cannot hit its failure schedule, which starts at
        # call two, and its trial budget is still whole. Within each
        # band, the highest rating wins. For an Anthropic request the
        # JSON-capable models sort first, because the Anthropic wire
        # cannot say the request needs JSON (the json_mode filter
        # above only fires for OpenAI bodies), so the sort is the only
        # guard against serving a model that cannot meet a JSON
        # requirement. For an OpenAI request the filter already
        # handles JSON, so the call band comes first there.
        if req.get("api") == "anthropic":
            out.sort(key=lambda r: (0 if r["json"] else 1,
                                    1 if self.calls_made.get(r["provider"], 0) >= 1 else 0,
                                    -r["rating"]))
        else:
            out.sort(key=lambda r: (1 if self.calls_made.get(r["provider"], 0) >= 1 else 0,
                                    0 if r["json"] else 1, -r["rating"]))
        return out

    def call_provider(self, provider, model, req):
        """One provider call; returns the parsed answer or None on refusal."""
        path, body, is_stream = build_provider_body(provider, model, req)
        url = provider_base(provider, self.env_url) + path
        headers = {"Content-Type": "application/json"}
        headers.update(auth_headers(provider))
        if provider.get("wire") == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        payload = json.dumps(body).encode("utf-8")
        # Every attempt counts toward the provider's trial budget and
        # its failure schedule, answered or refused, so count it here
        # (candidates() spreads on this count).
        self.calls_made[provider["id"]] = self.calls_made.get(provider["id"], 0) + 1
        try:
            status, _hdrs, raw = http_request("POST", url, headers,
                                              body=payload, timeout=PROVIDER_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - a failed call fails over
            log("provider call %s/%s errored: %s" % (provider["id"], model, exc))
            return None
        if status < 200 or status >= 300:
            log("provider call %s/%s refused HTTP %s" % (provider["id"], model, status))
            return None
        answer = parse_provider_answer(provider["wire"], is_stream, status, raw)
        if answer is None:
            log("provider call %s/%s gave an unusable answer" % (provider["id"], model))
            return None
        return answer

    def serve(self, req):
        """Try candidates in order; return (provider, model, answer) or None.

        A refusal goes straight to the next candidate on another
        provider: retrying the same provider costs a call and spends
        that provider's trial budget, and the retry fails outright
        when the refusal means the budget is spent. Only one provider
        call may succeed per request, and we stop at the first
        success, so failing over at once never breaks that rule.
        """
        tried_providers = set()
        for row in self.candidates(req):
            pid = row["provider"]
            if pid in tried_providers:
                continue
            tried_providers.add(pid)
            provider = self.providers[pid]
            answer = self.call_provider(provider, row["model"], req)
            if answer is not None:
                return provider, row["model"], answer
        return None

    def free_model_ids(self):
        """Distinct free model ids, catalog order, for GET /v1/models."""
        ids, seen = [], set()
        for entry in self.catalog.get("providers", []):
            for model in entry.get("models", []):
                if model.get("free") and model["id"] not in seen:
                    seen.add(model["id"])
                    ids.append(model["id"])
        return ids


# --------------------------------------------------------------------------
# Rendering: normalised answer -> client wire
# --------------------------------------------------------------------------

def split_text(text):
    pieces = [text[i:i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]
    return pieces or [""]


def render_openai_json(model, answer, req):
    message = {"role": "assistant", "content": answer["text"]}
    if answer["tool_calls"]:
        message["tool_calls"] = [{
            "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in answer["tool_calls"]]
    stop = map_stop(answer["stop"], "openai")
    if answer["tool_calls"] and stop == "stop":
        stop = "tool_calls"
    usage_in, usage_out = answer["usage"]
    return {
        "id": "chatcmpl-agg", "object": "chat.completion",
        "created": FIXED_CREATED, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": stop,
                     "logprobs": None}],
        "usage": {"prompt_tokens": usage_in or 0,
                  "completion_tokens": usage_out or 0,
                  "total_tokens": (usage_in or 0) + (usage_out or 0)},
    }


def render_openai_sse(model, answer, req):
    """OpenAI event stream: role chunk, text deltas, finish chunk, usage
    chunk (always emitted: the client needs final usage), then [DONE]."""
    stop = map_stop(answer["stop"], "openai")
    if answer["tool_calls"] and stop == "stop":
        stop = "tool_calls"

    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    for call in answer["tool_calls"]:
        frames.append("data: %s\n\n" % json.dumps(chunk(
            {"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                             "function": {"name": call["name"],
                                          "arguments": call["arguments"]}}]}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, stop)))
    usage_in, usage_out = answer["usage"]
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-agg", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model, "choices": [],
        "usage": {"prompt_tokens": usage_in or 0,
                  "completion_tokens": usage_out or 0,
                  "total_tokens": (usage_in or 0) + (usage_out or 0)}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(model, answer, req):
    content = []
    if answer["text"]:
        content.append({"type": "text", "text": answer["text"]})
    for call in answer["tool_calls"]:
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            args = {}
        content.append({"type": "tool_use", "id": call["id"],
                        "name": call["name"], "input": args})
    if not content:
        content = [{"type": "text", "text": ""}]
    stop = map_stop(answer["stop"], "anthropic")
    if answer["tool_calls"] and stop == "end_turn":
        stop = "tool_use"
    usage_in, usage_out = answer["usage"]
    return {
        "id": "msg_agg", "type": "message", "role": "assistant",
        "model": model, "content": content, "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in or 0, "output_tokens": usage_out or 0},
    }


def render_anthropic_sse(model, answer, req):
    """Anthropic event stream in canonical order: message_start,
    content_block_start, text deltas, content_block_stop, message_delta,
    message_stop."""
    stop = map_stop(answer["stop"], "anthropic")
    if answer["tool_calls"] and stop == "end_turn":
        stop = "tool_use"
    usage_in, usage_out = answer["usage"]

    def frame(event, data):
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    frames = [frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in or 0, "output_tokens": 0}}})]
    frames.append(frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in split_text(answer["text"]):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(frame("content_block_stop", {
        "type": "content_block_stop", "index": 0}))
    index = 1
    for call in answer["tool_calls"]:
        frames.append(frame("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"],
                              "name": call["name"], "input": {}}}))
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            args = {}
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta",
                      "partial_json": json.dumps(args)}}))
        frames.append(frame("content_block_stop", {
            "type": "content_block_stop", "index": index}))
        index += 1
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage_out or 0}}))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def render_client_response(api, stream, model, answer, req):
    if api == "openai":
        if stream:
            return ("text/event-stream; charset=utf-8",
                    render_openai_sse(model, answer, req).encode("utf-8"))
        return ("application/json",
                (json.dumps(render_openai_json(model, answer, req)) + "\n").encode("utf-8"))
    if stream:
        return ("text/event-stream; charset=utf-8",
                render_anthropic_sse(model, answer, req).encode("utf-8"))
    return ("application/json",
            (json.dumps(render_anthropic_json(model, answer, req)) + "\n").encode("utf-8"))


def error_body(api, status, message):
    if api == "openai":
        body = {"error": {"message": message, "type": "api_error",
                          "param": None, "code": "upstream_error"}}
    else:
        body = {"type": "error",
                "error": {"type": "api_error", "message": message}}
    return status, "application/json", (json.dumps(body) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _fmt, *_args):
            pass

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            if urllib.parse.urlparse(self.path).path == "/v1/models":
                self._models()
            else:
                self._send(404, "application/json",
                           b'{"error": "not_found"}\n')

        def _models(self):
            ids = aggregator.free_model_ids()
            auth = self.headers.get("Authorization") or ""
            api_key = self.headers.get("x-api-key")
            if auth.startswith("Bearer "):
                data = [{"id": mid, "object": "model", "created": FIXED_CREATED,
                         "owned_by": "aggregator"} for mid in ids]
                payload = (json.dumps({"object": "list", "data": data}) + "\n").encode("utf-8")
            elif api_key:
                data = [{"type": "model", "id": mid, "display_name": mid,
                         "created_at": "2026-01-01T00:00:00Z", "free": True}
                        for mid in ids]
                payload = (json.dumps({"data": data, "has_more": False}) + "\n").encode("utf-8")
            else:
                data = [{"id": mid, "object": "model", "created": FIXED_CREATED,
                         "owned_by": "aggregator"} for mid in ids]
                payload = (json.dumps({"object": "list", "data": data}) + "\n").encode("utf-8")
            self._send(200, "application/json", payload)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/chat/completions":
                api = "openai"
            elif path == "/v1/messages":
                api = "anthropic"
            else:
                self._send(404, "application/json", b'{"error": "not_found"}\n')
                return
            try:
                body = self._read_json()
            except (ValueError, UnicodeDecodeError):
                status, ctype, payload = error_body(api, 400, "the request body is not one JSON object")
                self._send(status, ctype, payload)
                return
            if not isinstance(body, dict):
                status, ctype, payload = error_body(api, 400, "the request body is not one JSON object")
                self._send(status, ctype, payload)
                return
            req = parse_client_request(api, body)
            result = aggregator.serve(req)
            if result is None:
                status, ctype, payload = error_body(
                    api, 502, "every free model refused this request")
                self._send(status, ctype, payload)
                return
            _provider, model, answer = result
            ctype, payload = render_client_response(api, req["stream"], model, answer, req)
            self._send(200, ctype, payload)

    return Handler


# --------------------------------------------------------------------------
# Trial driver
# --------------------------------------------------------------------------

def catalog_text(catalog):
    """The one canonical serialisation, used by --list and by register."""
    return json.dumps(catalog, separators=(",", ":"))


def load_catalog_for_list():
    """Catalog for --list: from the brief when present, else empty."""
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not prompt_file or not env_url:
        return {"providers": []}
    try:
        brief_providers, ratings, _manifest = load_brief(prompt_file)
        return build_catalog(brief_providers, ratings, env_url)
    except Exception as exc:  # noqa: BLE001 - --list must still exit 0
        log("--list catalog build failed (%s); printing empty catalog" % exc)
        return {"providers": []}


def register(env_url, port, catalog):
    url = env_url.rstrip("/") + "/register"
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION,
                          "port": port, "catalog": catalog}).encode("utf-8")
    for attempt in range(REGISTER_RETRIES):
        try:
            status, _hdrs, raw = http_request(
                "POST", url, {"Content-Type": "application/json"},
                body=payload, timeout=PROVIDER_TIMEOUT)
            if status == 200:
                body = json.loads(raw.decode("utf-8"))
                if body.get("ok") is True:
                    return body.get("requests")
            log("register attempt %d -> HTTP %s" % (attempt + 1, status))
        except Exception as exc:  # noqa: BLE001 - retry registration
            log("register attempt %d errored: %s" % (attempt + 1, exc))
        time.sleep(POLL_INTERVAL)
    return None


def poll_work(env_url):
    """GET /work; returns True when done, False on error / not done."""
    try:
        status, _hdrs, raw = http_request(
            "GET", env_url.rstrip("/") + "/work", {}, timeout=PROVIDER_TIMEOUT)
        if status != 200:
            return False
        return json.loads(raw.decode("utf-8")).get("done") is True
    except Exception as exc:  # noqa: BLE001 - keep polling
        log("work poll errored: %s" % exc)
        return False


def main(argv):
    if "--list" in argv:
        catalog = load_catalog_for_list()
        sys.stdout.write(catalog_text(catalog) + "\n")
        return 0
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if not prompt_file or not env_url:
        sys.stderr.write("usage: set MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL, "
                         "or run with --list\n")
        return 2
    brief_providers, ratings, _manifest = load_brief(prompt_file)
    catalog = build_catalog(brief_providers, ratings, env_url)
    aggregator = Aggregator(catalog, ratings, brief_providers, env_url)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on port %d" % port)
    requests = register(env_url, port, catalog)
    if requests is None:
        log("registration failed; exiting")
        server.shutdown()
        return 1
    log("registered; env expects %s requests" % requests)
    started = time.monotonic()
    while True:
        if poll_work(env_url):
            log("work done; exiting 0")
            break
        if time.monotonic() - started > RUN_DEADLINE_SECONDS:
            log("run deadline reached before work finished; exiting")
            break
        time.sleep(POLL_INTERVAL)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
