#!/usr/bin/env python3
"""Offline fake Anthropic Messages API for sandbox containment tests (no real model call).

  python3 scripts/fake_anthropic_api.py PORT STEPS.json LOG.txt

Each streamed request for the main model gets the next scripted tool call from STEPS.json
([[tool_name, input], ...]); after the last step it answers "done". Other models get "ok".
Point Claude Code at it with ANTHROPIC_BASE_URL=http://127.0.0.1:PORT ANTHROPIC_API_KEY=fake.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, STEPS, LOG = int(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
MAIN = "claude-opus-5-5"
state = {"i": 0}


def sse(ev, data):
    return f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(b"{}")

    do_HEAD = do_GET

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        model, stream = body.get("model", ""), body.get("stream", False)
        with open(LOG, "a") as f:
            f.write(f"POST {self.path} model={model} stream={stream}\n")
        if "count_tokens" in self.path:
            return self._json({"input_tokens": 10})
        block = {"type": "text", "text": "ok"}
        stop = "end_turn"
        if model.startswith(MAIN) and body.get("tools"):
            i = state["i"]; state["i"] += 1
            if i < len(STEPS):
                name, inp = STEPS[i]
                block = {"type": "tool_use", "id": f"toolu_fake{i}", "name": name, "input": inp}
                stop = "tool_use"
            else:
                block = {"type": "text", "text": "done"}
        msg = {"id": f"msg_fake{state['i']}", "type": "message", "role": "assistant", "model": model,
               "content": [], "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 1, "output_tokens": 1}}
        if not stream:
            msg.update(content=[block], stop_reason=stop)
            return self._json(msg)
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        out = [sse("message_start", {"type": "message_start", "message": msg})]
        if block["type"] == "tool_use":
            out.append(sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                   "content_block": {**block, "input": {}}}))
            out.append(sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "input_json_delta",
                                                             "partial_json": json.dumps(block["input"])}}))
        else:
            out.append(sse("content_block_start", {"type": "content_block_start", "index": 0,
                                                   "content_block": {"type": "text", "text": ""}}))
            out.append(sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "text_delta", "text": block["text"]}}))
        out.append(sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
        out.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                         "usage": {"output_tokens": 1}}))
        out.append(sse("message_stop", {"type": "message_stop"}))
        self.wfile.write(b"".join(out)); self.wfile.flush()

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
